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
import hashlib
import importlib.util
import itertools
import os
import pickle
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time

# Disable creation of __pycache__/.pyc files from rules.py files
sys.dont_write_bytecode = True

rules = {}
make_db = {} # XXX Accesses are not threadsafe right now
normpath_cache = {} # XXX Accesses are not threadsafe right now, though this only matters for msvc_show_includes
task_queue = queue.PriorityQueue()
event_queue = queue.Queue()
priority_queue_counter = itertools.count() # tiebreaker counter to fall back to FIFO when rule priorities are the same
any_errors = False

try:
    usable_columns = os.get_terminal_size().columns - 1 # avoid last column to prevent line wrap
except OSError:
    usable_columns = None # stdout is not attached to a terminal
show_progress_line = usable_columns is not None

def stdout_write(x):
    sys.stdout.write(x)
    sys.stdout.flush() # always flush log writes immediately

# Query existence and modification time in one stat() call for better performance.
def get_timestamp_if_exists(path):
    try:
        return os.stat(path).st_mtime
    except FileNotFoundError:
        return -1.0 # sentinel value: file does not exist

def normpath(path):
    if path in normpath_cache:
        return normpath_cache[path]
    ret = os.path.normpath(path)
    if os.name == 'nt':
        ret = ret.lower().replace('\\', '/')
    normpath_cache[path] = ret
    return ret

if os.name == 'nt': # evaluate this condition only once, rather than per call, for performance
    def joinpath(cwd, path):
        return path if (path[0] == '/' or path[1] == ':') else f'{cwd}/{path}'
else:
    def joinpath(cwd, path):
        return path if path[0] == '/' else f'{cwd}/{path}'

def execute(rule, verbose):
    # Run command, capture/filter its output, and get its exit code.
    # XXX Do we want to add an additional check that all the targets must exist?
    try:
        # Historical note: before Python 3.4 on Windows, subprocess.Popen() calls could inherit unrelated file handles
        # from other threads, leading to very strange file locking errors.  Fixed by: https://peps.python.org/pep-0446/
        p = subprocess.Popen(rule.cmd, cwd=rule.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except Exception as e:
        p = None
        out = str(e)
        code = 1
    if p is not None:
        out = p.stdout.read().decode().strip() # XXX What encoding should we use here??  This assumes UTF-8
        code = p.wait()
    if rule.msvc_show_includes:
        deps = set()
        r = re.compile('^Note: including file:\\s*(.*)$')
        new_out = []
        for line in out.splitlines():
            m = r.match(line)
            if m:
                dep = normpath(m.group(1))
                if not dep.startswith('c:/program files'):
                    deps.add(dep)
            else:
                new_out.append(line)
        with open(rule.depfile, 'w') as f:
            assert len(rule.targets) == 1
            f.write(f'{rule.targets[0]}: \\\n')
            for dep in sorted(deps):
                f.write(f'  {dep} \\\n')
            f.write('\n')

        # In addition to filtering out the /showIncludes messages, filter the one remaining
        # line of output where it just prints the source file name
        if len(new_out) == 1:
            out = ''
        else:
            out = '\n'.join(new_out)
    elif rule.output_exclude:
        r = re.compile(rule.output_exclude)
        out = '\n'.join(line for line in out.splitlines() if not r.match(line))

    built_text = 'Built %s.\n' % '\n  and '.join(repr(t) for t in rule.targets)
    if show_progress_line: # need to precede "Built [...]" with erasing the current progress indicator
        built_text = '\r%s\r%s' % (' ' * usable_columns, built_text)

    if verbose or code:
        if os.name == 'nt':
            quoted_cmd = subprocess.list2cmdline(rule.cmd)
        else:
            quoted_cmd = ' '.join(shlex.quote(x) for x in rule.cmd)
        out = f'{quoted_cmd}\n{out}'.rstrip()
    if code:
        global any_errors
        any_errors = True
        event_queue.put(('log', f'{built_text}{out}\n\n'))
        for t in rule.targets:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(t)
        return False

    local_make_db = make_db[rule.cwd]
    signature = rule.signature()
    for t in rule.targets:
        local_make_db[t] = signature
    if out:
        event_queue.put(('log', f'{built_text}{out}\n\n'))
    elif not show_progress_line:
        event_queue.put(('log', built_text))
    return True

class BuilderThread(threading.Thread):
    def __init__(self, verbose):
        super().__init__()
        self.verbose = verbose

    def run(self):
        while not any_errors:
            (priority, counter, rule) = task_queue.get()
            if rule is None:
                break
            if rule.cmd is not None:
                event_queue.put(('start', rule))
                execute(rule, self.verbose)
            event_queue.put(('finish', rule))

def schedule(target, visited, enqueued, completed):
    if target in visited or target in completed:
        return
    if target not in rules:
        visited.add(target)
        completed.add(target)
        return
    rule = rules[target]
    visited.update(rule.targets)
    if target in enqueued:
        return

    # Recurse into dependencies and order-only deps and wait for them to complete
    # Never recurse into depfile deps here, as the .d file could be stale/garbage from a previous build
    deps = [normpath(joinpath(rule.cwd, x)) for x in rule.deps]
    for dep in itertools.chain(deps, rule.order_only_inputs):
        schedule(dep, visited, enqueued, completed)
    if any(dep not in completed for dep in itertools.chain(deps, rule.order_only_inputs)):
        return

    # Error if any of the deps does not exist -- they should always exist by this point
    dep_timestamps = [get_timestamp_if_exists(dep) for dep in deps]
    for (dep, dep_timestamp) in zip(deps, dep_timestamps):
        if dep_timestamp < 0:
            error_message = f"ERROR: dependency {dep!r} of {' '.join(repr(t) for t in rule.targets)} is nonexistent\n"
            if show_progress_line:
                error_message = '\r%s\r%s' % (' ' * usable_columns, error_message)
            stdout_write(error_message)
            global any_errors
            any_errors = True
            exit(1)

    # Do all targets exist, and are all of them at least as new as every single dep?
    local_make_db = make_db[rule.cwd]
    target_timestamp = min(get_timestamp_if_exists(t) for t in rule.targets) # oldest target timestamp, or -1.0 if any target is nonexistent
    if target_timestamp >= 0 and all(dep_timestamp <= target_timestamp for dep_timestamp in dep_timestamps):
        # Is the rule's signature identical to the last time we ran it?
        signature = rule.signature()
        if all(local_make_db.get(t) == signature for t in rule.targets):
            # Parse the depfile, if present
            depfile_deps = []
            if rule.depfile:
                try:
                    with open(rule.depfile) as f:
                        depfile_deps = f.read()
                    depfile_deps = depfile_deps.replace('\\\n', '')
                    if '\\' in depfile_deps: # shlex.split is slow, don't use it unless we really need it
                        depfile_deps = shlex.split(depfile_deps)[1:]
                    else:
                        depfile_deps = depfile_deps.split()[1:]
                    depfile_deps = [normpath(joinpath(rule.cwd, x)) for x in depfile_deps]
                except FileNotFoundError:
                    depfile_deps = None # depfile was expected but missing -- always dirty

            # Do all depfile_deps exist, and are all targets at least as new as every single depfile_dep?
            if depfile_deps is not None and all(0 <= get_timestamp_if_exists(dep) <= target_timestamp for dep in depfile_deps):
                completed.update(rule.targets)
                return # skip the build

    # Remove stale targets immediately once this rule is marked dirty
    for t in rule.targets:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(t)
        local_make_db.pop(t, None) # XXX Probably better to store a None tombstone here for lock-free updates

    # Ensure targets' parent directories exist
    for t in rule.targets:
        os.makedirs(os.path.dirname(t), exist_ok=True)

    # Enqueue this task to a builder thread -- note that PriorityQueue needs the sense of priority reversed
    task_queue.put((-rule.priority, next(priority_queue_counter), rule))
    enqueued.update(rule.targets)

class Rule:
    def __init__(self, targets, deps, cwd, cmd, depfile, order_only_inputs, msvc_show_includes, output_exclude, latency):
        self.targets = targets
        self.deps = deps
        self.cwd = cwd
        self.cmd = cmd
        self.depfile = depfile
        self.order_only_inputs = order_only_inputs
        self.msvc_show_includes = msvc_show_includes
        self.output_exclude = output_exclude
        self.latency = latency
        self.priority = 0

    # order_only_inputs, output_exclude, priority are excluded from signatures because none of them should affect the targets' new content.
    def signature(self):
        info = (self.targets, self.deps, self.cwd, self.cmd, self.depfile, self.msvc_show_includes)
        return hashlib.sha256(pickle.dumps(info, protocol=4)).hexdigest() # XXX bump to protocol=5 once we drop 3.6/3.7 support

class BuildContext:
    def rule(self, outputs, inputs, *, cmd=None, depfile=None, order_only_inputs=None, msvc_show_includes=False, output_exclude=None, latency=1):
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

        rule = Rule(outputs, inputs, cwd, cmd, depfile, order_only_inputs, msvc_show_includes, output_exclude, latency)
        for t in outputs:
            if t in rules:
                print(f'ERROR: multiple ways to build {t!r}')
                exit(1)
            rules[t] = rule

# Reject disallowed constructs in rules.py -- a non-Turing-complete Starlark-like DSL
def validate_rules_ast(tree, path):
    BANNED = (
        ast.While, ast.Lambda, # prevent infinite loops and infinite recursion
        ast.ImportFrom, # XXX ast.Import temporarily allowed here as a transitional aid for now
        ast.With, ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
        ast.Global, ast.Nonlocal, ast.NamedExpr, ast.ClassDef,
        ast.Try, ast.Raise, ast.Yield, ast.YieldFrom, ast.Await,
        ast.Delete,
    )

    for node in ast.walk(tree):
        lineno = getattr(node, 'lineno', '?')
        if isinstance(node, BANNED):
            raise SyntaxError(f'{type(node).__name__} not allowed in rules.py (file {path!r}, line {lineno})')
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in {'os', 'platform'}:
                    raise SyntaxError(f'Import of {alias.name!r} not allowed (file {path!r}, line {lineno})')
        if isinstance(node, ast.Constant) and isinstance(node.value, (bytes, complex, float)):
            raise SyntaxError(f'{type(node.value).__name__} literal not allowed in rules.py (file {path!r}, line {lineno})')

def parse_rules_py(ctx, verbose, pathname, visited):
    if pathname in visited:
        return
    visited.add(pathname)
    if verbose:
        print(f'Parsing {pathname!r}...')

    with open(pathname, encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename=pathname)
    validate_rules_ast(tree, pathname)

    spec = importlib.util.spec_from_file_location(f'rules{len(visited)}', pathname)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {pathname!r}')
    rules_py_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules_py_module)

    dir = os.path.dirname(pathname)
    if dir not in make_db:
        make_db[dir] = {}
        path = f'{dir}/_out/make.db'
        with contextlib.suppress(FileNotFoundError):
            with open(path) as f:
                for line in f:
                    (target, signature) = line.rstrip().rsplit(' ', 1)
                    make_db[dir][target] = signature
    if hasattr(rules_py_module, 'submakes'):
        for f in rules_py_module.submakes():
            parse_rules_py(ctx, verbose, normpath(joinpath(dir, f)), visited)
    if hasattr(rules_py_module, 'rules'):
        ctx.cwd = dir
        rules_py_module.rules(ctx)

def propagate_latencies(target, latency, _active):
    if target in _active:
        print(f'ERROR: cycle detected involving {target!r}')
        exit(1)
    rule = rules[target]
    latency += rule.latency
    if latency <= rule.priority:
        return # nothing to do -- we are not increasing the priority of this rule
    rule.priority = latency # update this rule's latency

    # Recursively handle the dependencies, including order-only deps
    _active.add(target)
    deps = [normpath(joinpath(rule.cwd, x)) for x in rule.deps]
    for dep in itertools.chain(deps, rule.order_only_inputs):
        if dep in rules:
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

    # Set up rule DB, reading in make.db files as we go
    ctx = BuildContext()
    visited = set()
    for f in args.files:
        parse_rules_py(ctx, args.verbose, normpath(joinpath(cwd, f)), visited)
    for target in args.targets:
        if target not in rules:
            print(f'ERROR: no rule to build target {target!r}')
            exit(1)
        propagate_latencies(target, 0, set())

    # Clean up stale targets from previous builds that no longer have rules; also do an explicitly requested clean
    for (cwd, db) in make_db.items():
        if args.clean:
            dir = f'{cwd}/_out'
            if os.path.exists(dir):
                print(f'Cleaning {dir!r}...')
                shutil.rmtree(dir)
            db.clear()
        for (target, signature) in list(db.items()):
            if target not in rules:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(target)
                    print(f'Deleted stale target {target!r}.')
                del db[target]

    # Create and start builder threads
    threads = [BuilderThread(args.verbose) for i in range(args.jobs)]
    for t in threads:
        t.daemon = True # XXX this should probably be removed, but make sure Ctrl-C handling is correct
        t.start()

    # Do the build, and try to shut down as cleanly as possible if we get a Ctrl-C
    try:
        enqueued = set()
        completed = set()
        building = set()
        while True:
            # Enqueue work to the builders
            visited = set()
            for target in args.targets:
                schedule(target, visited, enqueued, completed)
            if all(target in completed for target in args.targets):
                break

            # Handle events from builder threads, then show progress update and exit if done
            # Be careful about iterating over data structures being edited concurrently by the BuilderThreads
            for (status, info) in drain_event_queue():
                if status == 'log':
                    stdout_write(info)
                elif status == 'start':
                    building.update(info.targets)
                else:
                    assert status == 'finish', status
                    building.difference_update(info.targets)
                    completed.update(info.targets)
            if any_errors:
                break
            if show_progress_line:
                incomplete_count = sum(1 for x in (visited - completed) if x in rules)
                if incomplete_count:
                    progress = ' '.join(sorted(x.rsplit('/', 1)[-1] for x in building))
                    progress = f'make.py: {incomplete_count} left, building: {progress}'
                else:
                    progress = ''
                if len(progress) < usable_columns:
                    pad = usable_columns - len(progress)
                    progress += ' ' * pad # erase old contents
                    progress += '\b' * pad # put cursor back at end of line
                else:
                    progress = progress[0:usable_columns]
                stdout_write('\r' + progress)
            if all(target in completed for target in args.targets):
                break
    finally:
        # Shut down the system by sending sentinel tokens to all the threads
        for i in range(args.jobs):
            task_queue.put((1000000, 0, None)) # lower priority than any real rule
        for t in threads:
            t.join()

        # Write out the final make.db files
        # XXX May want to do this "occasionally" as the build is running?  (not too often to avoid a perf hit, but often
        # enough to avoid data loss)
        for (cwd, db) in make_db.items():
            with contextlib.suppress(FileExistsError):
                os.mkdir(f'{cwd}/_out')
            with open(f'{cwd}/_out/make.db.tmp', 'w') as f:
                for (target, signature) in db.items():
                    f.write(f'{target} {signature}\n')
            os.replace(f'{cwd}/_out/make.db.tmp', f'{cwd}/_out/make.db')

    if any_errors:
        exit(1)

if __name__ == '__main__':
    main()
