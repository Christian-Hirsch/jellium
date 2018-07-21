from jellium.utils import unwrap_path

def draw_bridges(bridges, size, fname='wigner.tex'):
    """Draw bridges in tikz
    # Arguments
        edges:list of bridges
        size: array denoting the size of the torus
        fname: name of output file
    """
    with open(fname,'w') as f:
        f.write('\\begin{tikzpicture};\n')
        f.write('\\clip (-{0},-{1}) rectangle ({0},{1});\n'.format(*(size/2)))
        
        for bridge in bridges:
            for edge in unwrap_path(bridge, size):
                p, q = edge
                f.write("\\draw ({0:.2f}, {1:.2f})--({2:.2f}, {3:.2f});\n".format(*p, *q))
        f.write('\\end{tikzpicture}')
