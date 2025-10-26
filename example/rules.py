def rules(ctx):
    c_files = ['main.c']

    o_files = []
    for file in c_files:
        o_file = '_out/%s' % file.replace('.c', '.o')
        d_file = o_file.replace('.o', '.d')
        ctx.rule(o_file, file, cmd=['gcc', '-o', o_file, '-c', file, '-MD'], d_file=d_file)
        o_files += [o_file]

    exe_file = '_out/hello'
    ctx.rule(exe_file, o_files, cmd=['gcc', '-o', exe_file] + o_files)
