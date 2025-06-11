import sympy as sym
import numpy as np


'''
Adapting the stencil helper from driver to work here.
'''


def get_points(h, stencil_method, r_gp):
    if stencil_method == 'diamond':

        points = []

        center_col = 0
        center_row = 0



        top = (center_row-r_gp, center_col)
        #width on top starts at 1
        width = 1

        for i in range(0-r_gp, 0+r_gp+1):

            each_side = int((width+1)/2 - 1)
            for j in range(0-each_side, 0+each_side+1):

                points.append((h*j, h*i))


            if i-0+r_gp < r_gp:
                width += 2
            else:
                if i-0+r_gp == 2*r_gp + 1:
                    break
                else:
                    width -= 2

        return np.array(points)

    elif stencil_method == 'square':

        points = []

        center_col = 0
        center_row = 0


        top = (center_row-r_gp, center_col)
        #width on top starts at 1
        width = 1

        for i in range(0-r_gp, 0+r_gp+1):                
            for j in range(0-r_gp, 0+r_gp+1):

                points.append((h*j, h*i))

        return np.array(points)

    elif stencil_method == 'blocky_diamond':
        points = []

        center_col = 0
        center_row = 0

        top = (center_row - r_gp, center_col)
        # width on top starts at 1
        width = 1

        for i in range(0 - r_gp, 0 + r_gp + 1):
            for j in range(0 - r_gp, 0 + r_gp + 1):

                isCorner = ((i == 0 - r_gp and j == 0 - r_gp) or
                            (i == 0 + r_gp and j == 0 - r_gp) or
                            (i == 0 - r_gp and j == 0 + r_gp) or
                            (i == 0 + r_gp and j == 0 + r_gp))

                # block diamond with r_gp =1 is square.
                if r_gp < 3:
                    if r_gp == 1 or not isCorner:
                        points.append((h * j, h * i))

                elif r_gp == 3:
                    # Include all points except the four corners
                    if not isCorner:
                        points.append((h * j, h * i))

                else:
                    print(f"blocky_diamond stencil not implemented for r_gp: {r_gp}")

    
        return points
    
    elif stencil_method == 'cross':
        points = []

        center_col = 0
        center_row = 0

        for i in range(0-r_gp, 0+r_gp+1):
            points.append((0, h*i))  # Vertical line

        for j in range(0-r_gp, 0+r_gp+1):
            if j != 0:  # Avoid duplicating the center point
                points.append((h*j, 0))  # Horizontal line

        return np.array(points)


    else:
        raise ValueError(f"Invalid stencil method: {stencil_method}")

                    


if __name__ == '__main__':
    h = sym.symbols('h')

    print(get_points(h, 'square', 1)) #checked by hand
    print(get_points(h, 'blocky_diamond', 2)) #checked by hand
    print(get_points(h, 'diamond', 2)) #checked by hand
