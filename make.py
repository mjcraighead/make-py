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
import contextlib
import functools
import hashlib
import importlib.util
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

try:
    usable_columns = os.get_terminal_size().columns - 1 # avoid last column to prevent line wrap
except OSError:
    usable_columns = None # stdout is not attached to a terminal
show_progress_line = usable_columns is not None

def stdout_write(x):
    sys.stdout.write(x)
    sys.stdout.flush() # always flush log writes immediately

def die(msg):
    print(msg)
    sys.exit(1)

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
    # XXX Do we want to add an additional check that all the targets must exist?
    try:
        # Historical note: before Python 3.4 on Windows, subprocess.Popen() calls could inherit unrelated file handles
        # from other threads, leading to very strange file locking errors.  Fixed by: https://peps.python.org/pep-0446/
        result = subprocess.run(task.cmd, cwd=task.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = result.stdout.decode('utf-8', 'replace').strip() # Assumes UTF-8, but robust if not -- XXX consider changing out to bytes
        code = result.returncode
    except Exception as e:
        out = str(e)
        code = 1
    if task.msvc_show_includes:
        # Parse MSVC /showIncludes output, skipping system headers
        r = re.compile(r'^Note: including file:\s*(.*)$')
        (deps, new_out) = (set(), [])
        for line in out.splitlines():
            m = r.match(line)
            if m:
                dep = normpath(m.group(1))
                if not dep.startswith('c:/program files'):
                    deps.add(dep)
            else:
                new_out.append(line)
        out = '' if len(new_out) == 1 else '\n'.join(new_out) # drop lone "source.c" line printed by MSVC

        # Write a make-style depfile listing all included headers
        tmp_path = f'{task.depfile}.tmp'
        parts = [f'{task.targets[0]}:'] + sorted(deps) # we checked for only 1 target at task declaration time
        open(tmp_path, 'w').write(' \\\n  '.join(parts) + '\n') # add line continuations and indentation
        os.replace(tmp_path, task.depfile)
    elif task.output_exclude:
        r = re.compile(task.output_exclude)
        out = '\n'.join(line for line in out.splitlines() if not r.match(line))

    built_text = 'Built %s.\n' % '\n  and '.join(repr(t) for t in task.targets)
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
        for t in task.targets:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(t)
        return False

    local_make_db = make_db[task.cwd]
    signature = task.signature()
    for t in task.targets:
        assert t in local_make_db, t # make sure slot is already allocated
        local_make_db[t] = signature
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

def schedule(target, visited, enqueued, completed):
    if target in visited or target in completed:
        return
    task = tasks[target]
    visited.update(task.targets)
    if task in enqueued:
        return

    # Recurse into dependencies and order-only deps and wait for them to complete
    # Never recurse into depfile deps here, as the .d file could be stale/garbage from a previous run
    deps = [normpath(joinpath(task.cwd, x)) for x in task.deps]
    for dep in itertools.chain(deps, task.order_only_deps):
        if dep in tasks:
            schedule(dep, visited, enqueued, completed)
        else:
            visited.add(dep)
            completed.add(dep)
    if any(dep not in completed for dep in itertools.chain(deps, task.order_only_deps)):
        return

    # Error if any of the deps does not exist -- they should always exist by this point
    dep_timestamps = [get_timestamp_if_exists(dep) for dep in deps]
    for (dep, dep_timestamp) in zip(deps, dep_timestamps):
        if dep_timestamp < 0:
            global any_tasks_failed
            any_tasks_failed = True
            msg = f"ERROR: dependency {dep!r} of {' '.join(repr(t) for t in task.targets)} is nonexistent"
            if show_progress_line:
                msg = '\r%s\r%s' % (' ' * usable_columns, msg)
            die(msg)

    # Do all targets exist, and are all of them at least as new as every single dep?
    local_make_db = make_db[task.cwd]
    target_timestamp = min(get_timestamp_if_exists(t) for t in task.targets) # oldest target timestamp, or -1.0 if any target is nonexistent
    if target_timestamp >= 0 and all(dep_timestamp <= target_timestamp for dep_timestamp in dep_timestamps):
        # Is the task's signature identical to the last time we ran it?
        signature = task.signature()
        if all(local_make_db.get(t) == signature for t in task.targets):
            # Parse the depfile, if present
            depfile_deps = []
            if task.depfile:
                try:
                    depfile_deps = open(task.depfile).read().replace('\\\n', '')
                    if '\\' in depfile_deps: # shlex.split is slow, don't use it unless we really need it
                        depfile_deps = shlex.split(depfile_deps)[1:]
                    else:
                        depfile_deps = depfile_deps.split()[1:]
                    depfile_deps = [normpath(joinpath(task.cwd, x)) for x in depfile_deps]
                except FileNotFoundError:
                    depfile_deps = None # depfile was expected but missing -- always dirty

            # Do all depfile_deps exist, and are all targets at least as new as every single depfile_dep?
            if depfile_deps is not None and all(0 <= get_timestamp_if_exists(dep) <= target_timestamp for dep in depfile_deps):
                completed.update(task.targets)
                return # skip the task

    # Remove stale targets immediately once this task is marked dirty
    for t in task.targets:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(t)
        assert t in local_make_db, t # make sure slot is already allocated
        local_make_db[t] = None

    # Ensure targets' parent directories exist
    for t in task.targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)

    # Enqueue this task to the worker threads -- note that PriorityQueue needs the sense of priority reversed
    task_queue.put((-task.priority, next(priority_queue_counter), task))
    enqueued.add(task)

class Task:
    def __init__(self, targets, deps, cwd, cmd, depfile, order_only_deps, msvc_show_includes, output_exclude, latency):
        self.targets = targets
        self.deps = deps
        self.cwd = cwd
        self.cmd = cmd
        self.depfile = depfile
        self.order_only_deps = order_only_deps
        self.msvc_show_includes = msvc_show_includes
        self.output_exclude = output_exclude
        self.latency = latency
        self.priority = 0

    # order_only_deps, output_exclude, priority are excluded from signatures because none of them should affect the targets' new content.
    def signature(self):
        info = (self.targets, self.deps, self.cwd, self.cmd, self.depfile, self.msvc_show_includes)
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
    # Note that the DSL exposes "outputs"/"inputs", but these are remapped to "targets"/"deps" inside the internals of this script for clarity.
    def task(self, outputs, inputs, *, cmd=None, depfile=None, order_only_inputs=None, msvc_show_includes=False, output_exclude=None, latency=1):
        cwd = self.cwd
        if not isinstance(outputs, list):
            assert isinstance(outputs, str) # we expect outputs to be either a str (a single output) or a list of outputs
            outputs = [outputs]
        if cmd is None: # phony rule -- no command -- XXX do we want to support phony rules with commands?
            assert all(o.startswith(':') for o in outputs), outputs # phony rule targets must start with :
            assert depfile is None, depfile # phony rules cannot have depfiles
            assert order_only_inputs is None, order_only_inputs # phony rules cannot have order_only_inputs
            assert msvc_show_includes == False, msvc_show_includes # phony rules cannot set msvc_show_includes
            assert output_exclude is None, output_exclude # phony rules cannot set output_exclude
            # XXX override latency = 0?
        else: # real rule -- has a command
            assert all(o.startswith('_out/') for o in outputs), outputs # real rule targets must start with _out/
            assert isinstance(cmd, list), cmd # real rules must have a command, which is an argv list
            assert all(isinstance(x, str) for x in cmd), cmd
            cmd = cmd.copy()
        outputs = [normpath(joinpath(cwd, x)) for x in outputs]
        if not isinstance(inputs, list):
            assert isinstance(inputs, str) # we expect inputs to be either a str (a single input) or a list of inputs
            inputs = [inputs]
        inputs = inputs.copy()
        if depfile is not None:
            assert isinstance(depfile, str) # we expect depfile to be ether None or a str (the path of the depfile)
            depfile = normpath(joinpath(cwd, depfile))
        if order_only_inputs is None:
            order_only_inputs = []
        assert isinstance(order_only_inputs, list)
        order_only_inputs = [normpath(joinpath(cwd, x)) for x in order_only_inputs]
        assert output_exclude is None or isinstance(output_exclude, str)
        if msvc_show_includes:
            assert len(outputs) == 1, outputs # we only support 1 target for msvc_show_includes

        task = Task(outputs, inputs, cwd, cmd, depfile, order_only_inputs, msvc_show_includes, output_exclude, latency)
        for t in outputs:
            if t in tasks:
                die(f'ERROR: multiple ways to build {t!r}')
            tasks[t] = task
            if t not in make_db[cwd]:
                make_db[cwd][t] = None # preallocate a slot for every possible target in the make_db before we launch the WorkerThreads

    rule = task # ctx.task is the canonical interface, ctx.rule provided for familiarity

# Reject disallowed constructs in rules.py -- a non-Turing-complete Starlark-like DSL
def validate_rules_ast(tree, path):
    BANNED = (
        ast.While, ast.Lambda, # prevent infinite loops and infinite recursion
        ast.ImportFrom, # XXX ast.Import temporarily allowed here as a transitional aid for now
        ast.With, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
        ast.Global, ast.Nonlocal, ast.Delete, ast.ClassDef,
        ast.Try, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await,
        getattr(ast, 'NamedExpr', ()), # ast.NamedExpr exists only in 3.8+
    )

    for node in ast.walk(tree):
        lineno = getattr(node, 'lineno', '?')
        if isinstance(node, BANNED):
            raise SyntaxError(f'{type(node).__name__} not allowed in rules.py (file {path!r}, line {lineno})')
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {'os', 'platform'}:
                    raise SyntaxError(f'Import of {alias.name!r} not allowed (file {path!r}, line {lineno})')
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex, float)): # note: small loophole on 3.6/3.7, which uses ast.Bytes/Num instead
            raise SyntaxError(f'{type(node.value).__name__} literal not allowed in rules.py (file {path!r}, line {lineno})')

def parse_rules_py(ctx, verbose, pathname, visited):
    if pathname in visited:
        return
    visited.add(pathname)

    if verbose:
        print(f'Parsing {pathname!r}...')
    source = open(pathname, encoding='utf-8').read()
    tree = ast.parse(source, filename=pathname)
    validate_rules_ast(tree, pathname)

    spec = importlib.util.spec_from_file_location(f'rules{len(visited)}', pathname)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {pathname!r}')
    rules_py_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules_py_module)

    dirname = os.path.dirname(pathname)
    if dirname not in make_db:
        make_db[dirname] = {}
        with contextlib.suppress(FileNotFoundError):
            make_db[dirname] = dict(line.rstrip().rsplit(' ', 1) for line in open(f'{dirname}/_out/.make.db'))
    if hasattr(rules_py_module, 'submakes'):
        for f in rules_py_module.submakes():
            parse_rules_py(ctx, verbose, normpath(joinpath(dirname, f)), visited)
    if hasattr(rules_py_module, 'rules'):
        ctx.cwd = dirname
        rules_py_module.rules(ctx)

def propagate_latencies(target, latency, _active):
    if target in _active:
        die(f'ERROR: cycle detected involving {target!r}')
    task = tasks[target]
    latency += task.latency
    if latency <= task.priority:
        return # nothing to do -- we are not increasing the priority of this task
    task.priority = latency # update this task's latency

    # Recursively handle the dependencies, including order-only deps
    _active.add(target)
    deps = [normpath(joinpath(task.cwd, x)) for x in task.deps]
    for dep in itertools.chain(deps, task.order_only_deps):
        if dep in tasks:
            propagate_latencies(dep, latency, _active)
    _active.remove(target)

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

def main():
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true', help='clean before building')
    parser.add_argument('-f', '--file', dest='files', action='append', help='specify the path to a rules.py file (default is "rules.py")', metavar='FILE')
    parser.add_argument('-j', '--jobs', action='store', type=int, help='specify the number of parallel jobs (defaults to one per CPU)')
    parser.add_argument('-v', '--verbose', action='store_true', help='print verbose build output')
    parser.add_argument('targets', nargs='*', help='targets to build')
    args = parser.parse_args()
    if args.jobs is None:
        args.jobs = os.cpu_count() # default to one job per CPU
    if args.files is None:
        args.files = ['rules.py'] # default to "-f rules.py"

    cwd = os.getcwd()
    args.targets = [normpath(joinpath(cwd, x)) for x in args.targets]

    # Set up task DB, reading in .make.db files as we go
    ctx = EvalContext()
    ctx.host = detect_host()
    ctx.path = FrozenNamespace(expanduser=os.path.expanduser) # XXX temporary hole permitted in our sandbox to allow tasks to access ~
    visited = set()
    for f in args.files:
        parse_rules_py(ctx, args.verbose, normpath(joinpath(cwd, f)), visited)
    for target in args.targets:
        if target not in tasks:
            die(f'ERROR: no rule to build target {target!r}')
        propagate_latencies(target, 0, set())

    # Clean up stale targets from previous runs that no longer have tasks; also do an explicitly requested clean
    for (cwd, db) in make_db.items():
        if args.clean:
            dirname = f'{cwd}/_out'
            if os.path.exists(dirname):
                print(f'Cleaning {dirname!r}...')
                shutil.rmtree(dirname)
            for t in db:
                db[t] = None
        for (target, signature) in list(db.items()):
            if target not in tasks and signature is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(target)
                    print(f'Deleted stale target {target!r}.')
                del db[target]

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
            for target in args.targets:
                schedule(target, visited, enqueued, completed)
            if all(target in completed for target in args.targets):
                break

            # Handle events from worker threads, then show progress update and exit if done
            # Be careful about iterating over data structures being edited concurrently by the WorkerThreads
            for (status, payload) in drain_event_queue():
                if status == 'log':
                    stdout_write(payload)
                elif status == 'start':
                    running.add(payload)
                else:
                    assert status == 'finish', status
                    running.discard(payload)
                    completed.update(payload.targets)
            if any_tasks_failed:
                break
            if show_progress_line:
                remaining_count = len((visited - completed) & tasks.keys())
                if remaining_count:
                    def format_task_outputs(task):
                        targets = [t.rsplit('/', 1)[-1] for t in task.targets]
                        return targets[0] if len(targets) == 1 else f"[{' '.join(sorted(targets))}]"
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
                stdout_write('\r' + progress)
            if all(target in completed for target in args.targets):
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
            db = {target: signature for (target, signature) in db.items() if signature is not None} # remove None tombstones
            if db:
                with contextlib.suppress(FileExistsError):
                    os.mkdir(f'{cwd}/_out')
                tmp_path = f'{cwd}/_out/.make.db.tmp'
                open(tmp_path, 'w').write(''.join(f'{target} {signature}\n' for (target, signature) in db.items()))
                os.replace(tmp_path, f'{cwd}/_out/.make.db')
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(f'{cwd}/_out/.make.db')

    if any_tasks_failed:
        sys.exit(1)

if __name__ == '__main__':
    main()
