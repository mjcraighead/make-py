def rules(ctx: str):
    # Source files
    c_files = ['main.c']

    # Compile each .c file into a .o with an accompanying depfile
    o_files = []
    for src in c_files:
        o_file = f"_out/{src.replace('.c', '.o')}"
        depfile = o_file.replace('.o', '.d')
        ctx.rule(o_file, src, cmd=['gcc', '-o', o_file, '-c', src, '-MD'], depfile=depfile)
        o_files.append(o_file)

    # Link all objects into the final executable
    exe_file = '_out/hello'
    ctx.rule(exe_file, o_files, cmd=['gcc', '-o', exe_file, *o_files])

    # Phony aggregate targets
    ctx.rule(':build', exe_file)
    ctx.rule(':all', ':build')
