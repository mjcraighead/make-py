#!/usr/bin/env python3
#
# make.py (https://github.com/mjcraighead/make-py)
# Copyright (c) 2012-2025 Matt Craighead
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import ast
import builtins
import contextlib
import functools
import hashlib
import importlib.util
import inspect
import itertools
import os
import pickle
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading

# Disable creation of __pycache__/.pyc files from rules.py files
sys.dont_write_bytecode = True

tasks = {}
make_db = {}
task_queue = queue.PriorityQueue()
event_queue = queue.Queue()
priority_queue_counter = itertools.count() # tiebreaker counter to fall back to FIFO when task priorities are the same
any_tasks_failed = False # global failure flag across all tasks in this run
default_subprocess_env = os.environ.copy() # default inherited env for subprocess.run

try:
    usable_columns = os.get_terminal_size().columns - 1 # avoid last column to prevent line wrap
except OSError:
    usable_columns = None # stdout is not attached to a terminal
show_progress_line = usable_columns is not None

def die(msg):
    print(msg)
    sys.exit(1)

def die_at(path, lineno, msg):
    die(f'ERROR: {os.path.relpath(path)}:{lineno}: {msg}')

def _expect(cond, path, lineno, msg):
    if not cond:
        die_at(path, lineno, msg)

# Query existence and modification time in one stat() call for better performance.
def get_timestamp_if_exists(path):
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return -1.0 # sentinel value: file does not exist

if os.name == 'nt': # evaluate this condition only once, rather than per call, for performance
    @functools.lru_cache(maxsize=None)
    def normpath(path):
        return os.path.normpath(path).lower().replace('\\', '/')

    def joinpath(cwd, path):
        return path if (path[0] == '/' or path[1:2] == ':') else f'{cwd}/{path}'
else:
    @functools.lru_cache(maxsize=None)
    def normpath(path):
        return os.path.normpath(path)

    def joinpath(cwd, path):
        return path if path[0] == '/' else f'{cwd}/{path}'

def execute(task, verbose):
    # Run command, capture/filter its output, and get its exit code.
    # XXX Do we want to add an additional check that all the outputs must exist?
    try:
        # Historical note: before Python 3.4 on Windows, subprocess.Popen() calls could inherit unrelated file handles
        # from other threads, leading to very strange file locking errors.  Fixed by: https://peps.python.org/pep-0446/
        result = subprocess.run(task.cmd, cwd=task.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=default_subprocess_env)
        out = result.stdout.decode('utf-8', 'replace').strip() # Assumes UTF-8, but robust if not -- XXX consider changing out to bytes
        code = result.returncode
    except Exception as e:
        out = str(e)
        code = 1
    if task.msvc_show_includes:
        # Parse MSVC /showIncludes output, skipping system headers
        r = re.compile(r'^Note: including file:\s*(.*)$')
        (inputs, new_out) = (set(), [])
        for line in out.splitlines():
            m = r.match(line)
            if m:
                input = normpath(m.group(1))
                if not input.startswith('c:/program files'):
                    inputs.add(input)
            else:
                new_out.append(line)
        out = '' if len(new_out) == 1 else '\n'.join(new_out) # drop lone "source.c" line printed by MSVC

        # Write a make-style depfile listing all included headers
        tmp_path = f'{task.depfile}.tmp'
        parts = [f'{task.outputs[0]}:', *sorted(inputs)] # we checked for only 1 output at task declaration time
        open(tmp_path, 'w').write(' \\\n  '.join(parts) + '\n') # add line continuations and indentation
        os.replace(tmp_path, task.depfile)
    elif task.output_exclude:
        r = re.compile(task.output_exclude)
        out = '\n'.join(line for line in out.splitlines() if not r.match(line))

    built_text = 'Built %s.\n' % '\n  and '.join(repr(output) for output in task.outputs)
    if show_progress_line: # need to precede "Built [...]" with erasing the current progress indicator
        built_text = '\r%s\r%s' % (' ' * usable_columns, built_text)

    if verbose or code:
        if os.name == 'nt':
            quoted_cmd = subprocess.list2cmdline(task.cmd)
        else:
            quoted_cmd = ' '.join(shlex.quote(x) for x in task.cmd)
        out = f'{quoted_cmd}\n{out}'.rstrip()
    if code:
        global any_tasks_failed
        any_tasks_failed = True
        event_queue.put(('log', f'{built_text}{out}\n\n'))
        for output in task.outputs:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(output)
        return False

    local_make_db = make_db[task.cwd]
    signature = task.signature()
    for output in task.outputs:
        assert output in local_make_db, output # make sure slot is already allocated
        local_make_db[output] = signature
    if out:
        event_queue.put(('log', f'{built_text}{out}\n\n'))
    elif not show_progress_line:
        event_queue.put(('log', built_text))
    return True

class WorkerThread(threading.Thread):
    def __init__(self, verbose):
        super().__init__()
        self.verbose = verbose

    def run(self):
        while not any_tasks_failed:
            (_, _, task) = task_queue.get()
            if task is None:
                break
            if task.cmd is not None:
                event_queue.put(('start', task))
                execute(task, self.verbose)
            event_queue.put(('finish', task))

# Note: external orchestrators may predeclare certain outputs as hermetically clean.
# make.py treats such declarations as axiomatic -- they come from elsewhere.
def schedule(output, visited, enqueued, completed):
    if output in visited or output in completed:
        return
    task = tasks[output]
    visited.update(task.outputs)
    if task in enqueued:
        return

    # Recurse into inputs and order-only inputs and wait for them to complete
    # Never recurse into depfile inputs here, as the .d file could be stale/garbage from a previous run
    for input in itertools.chain(task.inputs, task.order_only_inputs):
        if input in tasks:
            schedule(input, visited, enqueued, completed)
        else:
            visited.add(input)
            completed.add(input)
    if any(input not in completed for input in itertools.chain(task.inputs, task.order_only_inputs)):
        return

    # Error if any of the inputs does not exist -- they should always exist by this point
    input_timestamps = [get_timestamp_if_exists(input) for input in task.inputs]
    for (input, input_timestamp) in zip(task.inputs, input_timestamps):
        if input_timestamp < 0:
            if input not in tasks or tasks[input].cmd is not None: # source file or real (not phony) rule; do not check for phony rules
                global any_tasks_failed
                any_tasks_failed = True
                msg = f"ERROR: input {input!r} of {' '.join(repr(output) for output in task.outputs)} is nonexistent"
                if show_progress_line:
                    msg = '\r%s\r%s' % (' ' * usable_columns, msg)
                die(msg)

    # Do all outputs exist, and are all of them at least as new as every single input?
    local_make_db = make_db[task.cwd]
    output_timestamp = min(get_timestamp_if_exists(output) for output in task.outputs) # oldest output timestamp, or -1.0 if any output is nonexistent
    if output_timestamp >= 0 and all(input_timestamp <= output_timestamp for input_timestamp in input_timestamps):
        # Is the task's signature identical to the last time we ran it?
        signature = task.signature()
        if all(local_make_db.get(output) == signature for output in task.outputs):
            # Parse the depfile, if present
            depfile_inputs = []
            if task.depfile:
                try:
                    depfile_inputs = open(task.depfile, encoding='utf-8').read().replace('\\\n', '')
                    if '\\' in depfile_inputs: # shlex.split is slow, don't use it unless we really need it
                        depfile_inputs = shlex.split(depfile_inputs)[1:]
                    else:
                        depfile_inputs = depfile_inputs.split()[1:]
                    depfile_inputs = [normpath(joinpath(task.cwd, x)) for x in depfile_inputs]
                except FileNotFoundError:
                    depfile_inputs = None # depfile was expected but missing -- always dirty
                except Exception: # anything else that went wrong
                    msg = f"WARNING: malformed depfile for {' '.join(repr(output) for output in task.outputs)} (will rebuild)"
                    if show_progress_line:
                        msg = '\r%s\r%s' % (' ' * usable_columns, msg)
                    print(msg)
                    depfile_inputs = None

            # Do all depfile_inputs exist, and are all outputs at least as new as every single depfile_input?
            if depfile_inputs is not None and all(0 <= get_timestamp_if_exists(input) <= output_timestamp for input in depfile_inputs):
                completed.update(task.outputs)
                return # skip the task

    # Remove stale outputs immediately once this task is marked dirty
    for output in task.outputs:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(output)
        assert output in local_make_db, output # make sure slot is already allocated
        local_make_db[output] = None

    # Ensure outputs' parent directories exist
    for output in task.outputs:
        os.makedirs(os.path.dirname(output), exist_ok=True)

    # Enqueue this task to the worker threads -- note that PriorityQueue needs the sense of priority reversed
    task_queue.put((-task.priority, next(priority_queue_counter), task))
    enqueued.add(task)

class Task:
    def __init__(self, outputs, inputs, cwd, cmd, depfile, order_only_inputs, msvc_show_includes, output_exclude, latency, path, lineno):
        self.outputs = outputs
        self.inputs = inputs
        self.cwd = cwd
        self.cmd = cmd
        self.depfile = depfile
        self.order_only_inputs = order_only_inputs
        self.msvc_show_includes = msvc_show_includes
        self.output_exclude = output_exclude
        self.latency = latency
        self.priority = 0
        self.path = path
        self.lineno = lineno

    # output_exclude and priority are excluded from signatures because they do not affect the outputs' content.
    # outputs, inputs, and order_only_inputs are included since they alter DAG structure (and thus execution ordering and correctness).
    def signature(self):
        info = (self.outputs, self.inputs, self.cwd, self.cmd, self.depfile, self.order_only_inputs, self.msvc_show_includes)
        return hashlib.sha256(pickle.dumps(info, protocol=4)).hexdigest() # XXX bump to protocol=5 once we drop 3.6/3.7 support

class FrozenNamespace:
    __slots__ = ('__dict__',) # allow exactly one slot: the instance dict itself
    def __init__(self, **kwargs):
        object.__setattr__(self, '__dict__', dict(kwargs))
    def __setattr__(self, k, v):
        raise AttributeError(f'{self.__class__.__name__} is read-only')
    __delattr__ = __setattr__
    def __repr__(self):
        items = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({items})'

# make.py host detection: runs on any plausible system in 2025 with Python 3.6+.
# ctx.host.os = OS ABI family (kernel/loader/libc), ctx.host.arch = CPU ISA family; together define the host ABI.
os_map = {
    'Windows': 'windows', 'Linux': 'linux', 'Darwin': 'darwin',
    'FreeBSD': 'freebsd', 'OpenBSD': 'openbsd', 'NetBSD': 'netbsd',
    'DragonFly': 'dragonflybsd', 'SunOS': 'sunos',
}
arch_map = {
    'AMD64': 'x86_64', 'x86_64': 'x86_64',
    'x86': 'x86_32', 'i686': 'x86_32',
    'ARM64': 'aarch64', 'aarch64': 'aarch64', 'arm64': 'aarch64',
    'ppc64le': 'ppc64le', 'riscv64': 'riscv64', 's390x': 's390x',
}
def detect_host():
    (system, machine) = (platform.system(), platform.machine())
    if system not in os_map or machine not in arch_map:
        die(f'ERROR: host detection failed: system={system!r} machine={machine!r}')
    return FrozenNamespace(os=os_map[system], arch=arch_map[machine])

class EvalContext:
    def task(self, outputs, inputs, *, cmd=None, depfile=None, order_only_inputs=None, msvc_show_includes=False, output_exclude=None, latency=1):
        frame = inspect.currentframe().f_back
        (path, lineno) = (frame.f_code.co_filename, frame.f_lineno)
        cwd = self.cwd
        if not isinstance(outputs, list):
            _expect(isinstance(outputs, str), path, lineno, 'outputs must be either a str or a list')
            outputs = [outputs]
        if cmd is None: # phony rule -- no command -- XXX do we want to support phony rules with commands?
            _expect(all(o.startswith(':') for o in outputs), path, lineno, 'phony rule outputs must start with :')
            _expect(not any('/' in o for o in outputs), path, lineno, 'phony rule outputs must not contain path separators')
            _expect(depfile is None, path, lineno, 'phony rules cannot have depfiles')
            _expect(order_only_inputs is None, path, lineno, 'phony rules cannot have order_only_inputs')
            _expect(msvc_show_includes == False, path, lineno, 'phony rules cannot set msvc_show_includes')
            _expect(output_exclude is None, path, lineno, 'phony rules cannot set output_exclude')
            latency = 0 # no command, therefore zero execution latency
        else: # real rule -- has a command
            _expect(all(o.startswith('_out/') for o in outputs), path, lineno, "rule output paths must start with '_out/'")
            _expect(isinstance(cmd, list) and all(isinstance(x, str) for x in cmd), path, lineno, 'real rules must set cmd=[argv_list]')
            cmd = cmd.copy()
        outputs = [normpath(joinpath(cwd, x)) for x in outputs]
        if not isinstance(inputs, list):
            _expect(isinstance(inputs, str), path, lineno, 'inputs must be either a str or a list')
            inputs = [inputs]
        inputs = [normpath(joinpath(cwd, x)) for x in inputs]
        if depfile is not None:
            _expect(isinstance(depfile, str), path, lineno, 'depfile must be either None or a str')
            depfile = normpath(joinpath(cwd, depfile))
        if order_only_inputs is None:
            order_only_inputs = []
        _expect(isinstance(order_only_inputs, list), path, lineno, 'order_only_inputs must be either None or a list')
        order_only_inputs = [normpath(joinpath(cwd, x)) for x in order_only_inputs]
        _expect(output_exclude is None or isinstance(output_exclude, str), path, lineno, 'output_exclude must be either None or a str')
        if msvc_show_includes:
            _expect(len(outputs) == 1, path, lineno, 'msvc_show_includes requires only a single output')

        task = Task(outputs, inputs, cwd, cmd, depfile, order_only_inputs, msvc_show_includes, output_exclude, latency, path, lineno)
        for output in outputs:
            if output in tasks:
                die(f'ERROR: multiple tasks declare {output!r}:\n'
                    f'  first declared at {os.path.relpath(tasks[output].path)}:{tasks[output].lineno}\n'
                    f'  again declared at {os.path.relpath(path)}:{lineno}')
            tasks[output] = task
            if output not in make_db[cwd]:
                make_db[cwd][output] = None # preallocate a slot for every possible output in the make_db before we launch the WorkerThreads

    rule = task # ctx.task is the canonical interface, ctx.rule provided for familiarity

# Reject disallowed constructs in tasks.py/rules.py -- a non-Turing-complete Starlark-like DSL
def validate_tasks_ast(tree, path):
    BANNED = (
        ast.While, ast.Lambda, # prevent infinite loops and infinite recursion
        ast.Import, ast.ImportFrom,
        ast.With, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
        ast.Global, ast.Nonlocal, ast.Delete, ast.ClassDef,
        ast.Try, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await,
        getattr(ast, 'NamedExpr', ()), # ast.NamedExpr exists only in 3.8+
    )

    for node in ast.walk(tree):
        lineno = getattr(node, 'lineno', '?')
        if isinstance(node, BANNED):
            die_at(path, lineno, f'{type(node).__name__} not allowed')
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex, float)): # note: small loophole on 3.6/3.7, which uses ast.Bytes/Num instead
            die_at(path, lineno, f'{type(node.value).__name__} literal not allowed')

CTX_FIELDS = ('host', 'env', 'path', 'task', 'rule', 'cwd')
SAFE_BUILTINS = (
    'len', 'range', 'print', 'repr', # essentials and debugging
    'enumerate', 'zip', 'sorted', 'reversed', # common iteration helpers
    'list', 'dict', 'set', 'tuple', 'frozenset', 'str', 'int', 'bool', # basic types/constructors
    'abs', 'min', 'max', 'sum', 'any', 'all', # math and logic
)
safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTINS}

def eval_tasks_py(ctx, verbose, pathname, index):
    if verbose:
        print(f'Parsing {pathname!r}...')
    source = open(pathname, encoding='utf-8').read()
    tree = ast.parse(source, filename=pathname)
    validate_tasks_ast(tree, pathname)

    spec = importlib.util.spec_from_file_location(f'tasks{index}', pathname)
    if spec is None or spec.loader is None:
        die(f'ERROR: cannot import {pathname!r}')
    tasks_py_module = importlib.util.module_from_spec(spec)
    tasks_py_module.__dict__['__builtins__'] = safe_builtins
    spec.loader.exec_module(tasks_py_module)

    dirname = os.path.dirname(pathname)
    if dirname not in make_db:
        make_db[dirname] = {}
        with contextlib.suppress(FileNotFoundError):
            make_db[dirname] = dict(line.rstrip().rsplit(' ', 1) for line in open(f'{dirname}/_out/.make.db'))
    ctx.cwd = dirname
    frozen_ctx = FrozenNamespace(**{k: getattr(ctx, k) for k in CTX_FIELDS})
    for name in ['tasks', 'rules']: # evaluate modern API first, then legacy API if present
        if hasattr(tasks_py_module, name):
            getattr(tasks_py_module, name)(frozen_ctx)

def locate_tasks_py_dir(path):
    for pattern in ['/_out/', '/:']: # look for standard and phony rules
        i = path.rfind(pattern)
        if i >= 0:
            return path[:i] # tasks.py lives in the parent directory
    return None

def discover_tasks(ctx, verbose, output, visited_files, visited_dirs, _active):
    if output in _active:
        die(f'ERROR: cycle detected involving {output!r}')
    if output in visited_files:
        return
    visited_files.add(output)

    # Locate and evaluate the tasks.py for this output (if we haven't already evaluated it)
    tasks_py_dir = locate_tasks_py_dir(output)
    if tasks_py_dir is None:
        return # this is a source file, not an output file or a phony rule name -- we are done
    if tasks_py_dir not in visited_dirs:
        tasks_py_path = f'{tasks_py_dir}/tasks.py'
        if not os.path.exists(tasks_py_path):
            tasks_py_path = f'{tasks_py_dir}/rules.py' # tasks.py is the canonical name, rules.py provided for familiarity
        eval_tasks_py(ctx, verbose, tasks_py_path, len(visited_dirs))
        visited_dirs.add(tasks_py_dir)

    if output not in tasks:
        die(f'ERROR: no rule to make {output!r}')
    task = tasks[output]
    _active.add(output)
    for input in itertools.chain(task.inputs, task.order_only_inputs):
        discover_tasks(ctx, verbose, input, visited_files, visited_dirs, _active)
    _active.remove(output)

def propagate_latencies(output, latency):
    task = tasks[output]
    latency += task.latency
    if latency <= task.priority:
        return # nothing to do -- we are not increasing the priority of this task
    task.priority = latency # update this task's latency and recurse
    for input in itertools.chain(task.inputs, task.order_only_inputs):
        if input in tasks:
            propagate_latencies(input, latency)

def drain_event_queue():
    while True:
        try:
            # Warning: this blocks KeyboardInterrupt during the timeout on Windows
            events = [event_queue.get(timeout=0.05 if os.name == 'nt' else None)]
            break
        except queue.Empty:
            continue # keep trying until we get at least one event (only hit on Windows)
    while True:
        try:
            events.append(event_queue.get_nowait())
        except queue.Empty:
            break
    return events

def parse_env_args(args):
    if os.name == 'nt': # Windows: inject the smallest viable subset of os.environ needed to execute system tools
        keys = ['ProgramFiles', 'ProgramFiles(x86)', 'CommonProgramFiles', 'CommonProgramFiles(x86)', 'SystemRoot', 'ComSpec',
                'TEMP', 'TMP', 'PATH', 'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE']
        env = {k: os.environ[k] for k in keys if k in os.environ}
    else:
        env = {} # POSIX: no injection; hermetic by default
    for arg in args:
        if '=' not in arg:
            die(f'ERROR: invalid --env format (expected key=value): {arg!r}')
        (k, v) = arg.split('=', 1)
        if not k.isidentifier():
            die(f'ERROR: invalid key name for --env: {k!r}')
        env[k] = v
    return FrozenNamespace(**env)

def minimal_env(ctx):
    if ctx.host.os == 'windows': # currently identical to parse_env_args above, but may diverge if more are needed
        keys = ['ProgramFiles', 'ProgramFiles(x86)', 'CommonProgramFiles', 'CommonProgramFiles(x86)', 'SystemRoot', 'ComSpec',
                'TEMP', 'TMP', 'PATH', 'NUMBER_OF_PROCESSORS', 'PROCESSOR_ARCHITECTURE']
        return {k: os.environ[k] for k in keys if k in os.environ}
    else:
        path = '/usr/local/bin:/usr/bin:/bin'
        if ctx.host.os == 'sunos':
            path = '/usr/xpg4/bin:' + path
        return {
            'PATH': path,
            'TMPDIR': '/tmp',
            'HOME': '/homeless-shelter',
            'USER': 'nobody',
            'SHELL': '/bin/sh',
            'LC_ALL': 'C.UTF-8',
            'LANG': 'C.UTF-8',
            'TZ': 'UTC',
            'SOURCE_DATE_EPOCH': '0',
        }

def main():
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true', help='clean _out directories first')
    parser.add_argument('-j', '--jobs', action='store', type=int, help='specify the number of parallel jobs (defaults to one per CPU)')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose output')
    parser.add_argument('--env', action='append', default=[], help='set ctx.env.KEY to VALUE in rules.py evaluation environment', metavar='KEY=VALUE')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--minimal-env', action='store_true', help='use a minimal hermetic environment for subprocesses (future default)')
    group.add_argument('--inherit-env', action='store_true', help='explicitly inherit full host environment in subprocesses (current default)')
    parser.add_argument('outputs', nargs='*', help='outputs to make')
    args = parser.parse_args()
    if args.jobs is None:
        args.jobs = os.cpu_count() # default to one job per CPU

    cwd = os.getcwd()
    args.outputs = [normpath(joinpath(cwd, x)) for x in args.outputs]

    # Set up EvalContext and task DB, reading in .make.db files as we go
    ctx = EvalContext()
    ctx.host = detect_host()
    ctx.env = parse_env_args(args.env)
    ctx.path = FrozenNamespace(expanduser=os.path.expanduser) # XXX temporary hole permitted in our sandbox to allow tasks to access ~
    if args.minimal_env: # use hermetic baseline instead of inherited environment
        global default_subprocess_env
        default_subprocess_env = minimal_env(ctx)
    (visited_files, visited_dirs) = (set(), set())
    for output in args.outputs:
        discover_tasks(ctx, args.verbose, output, visited_files, visited_dirs, set())
    for output in args.outputs:
        if output not in tasks:
            die(f'ERROR: no rule to make {output!r}')
        propagate_latencies(output, 0)

    # Clean up stale outputs from previous runs that no longer have tasks; also do an explicitly requested clean
    for (cwd, db) in make_db.items():
        if args.clean:
            dirname = f'{cwd}/_out'
            if os.path.exists(dirname):
                print(f'Cleaning {dirname!r}...')
                shutil.rmtree(dirname)
            for output in db:
                db[output] = None
        for (output, signature) in list(db.items()):
            if output not in tasks and signature is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(output)
                    print(f'Deleted stale output {output!r}.')
                del db[output]

    # Create and start worker threads
    threads = [WorkerThread(args.verbose) for i in range(args.jobs)]
    for t in threads:
        t.daemon = True # XXX this should probably be removed, but make sure Ctrl-C handling is correct
        t.start()

    # Main loop: schedule/execute tasks, report progress, and shut down as cleanly as possible if we get a Ctrl-C
    try:
        (enqueued, completed, running) = (set(), set(), set())
        while True:
            # Enqueue tasks to the workers
            visited = set()
            for output in args.outputs:
                schedule(output, visited, enqueued, completed)
            if all(output in completed for output in args.outputs):
                break

            # Handle events from worker threads, then show progress update and exit if done
            # Be careful about iterating over data structures being edited concurrently by the WorkerThreads
            for (status, payload) in drain_event_queue():
                if status == 'log':
                    sys.stdout.write(payload)
                    sys.stdout.flush()
                elif status == 'start':
                    running.add(payload)
                else:
                    assert status == 'finish', status
                    running.discard(payload)
                    completed.update(payload.outputs)
            if any_tasks_failed:
                break
            if show_progress_line:
                remaining_count = len((visited - completed) & tasks.keys())
                if remaining_count:
                    def format_task_outputs(task):
                        outputs = [output.rsplit('/', 1)[-1] for output in task.outputs]
                        return outputs[0] if len(outputs) == 1 else f"[{' '.join(sorted(outputs))}]"
                    names = ' '.join(sorted(format_task_outputs(task) for task in running))
                    progress = f'make.py: {remaining_count} left, building: {names}'
                else:
                    progress = ''
                if len(progress) < usable_columns:
                    pad = usable_columns - len(progress)
                    progress += ' ' * pad # erase old contents
                    progress += '\b' * pad # put cursor back at end of line
                else:
                    progress = progress[:usable_columns]
                sys.stdout.write('\r' + progress)
                sys.stdout.flush()
            if all(output in completed for output in args.outputs):
                break
    finally:
        # Shut down the system by sending sentinel tokens to all the threads
        for i in range(args.jobs):
            task_queue.put((1000000, 0, None)) # lower priority than any real task
        for t in threads:
            t.join()

        # Write out the final .make.db files
        # XXX May want to do this "occasionally" as tasks are running?  (not too often to avoid a perf hit, but often
        # enough to avoid data loss)
        for (cwd, db) in make_db.items():
            db = {output: signature for (output, signature) in db.items() if signature is not None} # remove None tombstones
            if db:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(f'{cwd}/_out')
                tmp_path = f'{cwd}/_out/.make.db.tmp'
                open(tmp_path, 'w').write(''.join(f'{output} {signature}\n' for (output, signature) in db.items()))
                os.replace(tmp_path, f'{cwd}/_out/.make.db')
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(f'{cwd}/_out/.make.db')

    if any_tasks_failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
# To those who understand what this really is: you already know where to look.
