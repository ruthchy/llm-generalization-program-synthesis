import re
class generateLOGOPseudoCode():
    def __init__(self):
        pass

#Basic Shapes
    def generate_line(self, length: int, angle: float = 0.0, left: bool = True) -> str:
        if not (0.0 <= angle <= 360.0):
            raise ValueError("Angle must be between 0.0 and 360.0")
        if angle == 0:
            return f"forward({length})"
        else:
            direction = "left" if left else "right"
            return f"forward({length})\n{direction}({angle})"
            
    def generate_polygon(self, sides: int, length: int, left: bool = True) -> str:
        angle = 360 / sides
        direction = "left" if left else "right"
        return f"for i in range({sides}):\n    forward({length})\n    {direction}({angle})"

    def generate_semicircle(self, size: int, semicircle: bool = True, left: bool = True) -> str:
        EPS_ANGLE = 1.0
        EPS_DIST = 0.03490481287456702  # Using the defined EPS_DIST value
        HALF_INF = 180
        size = EPS_DIST * size

        direction = "left" if left else "right"
        semicircle_code = f"for i in range({HALF_INF}):\n    forward({size})\n    {direction}({EPS_ANGLE})"
        
        if semicircle:
            return semicircle_code
        else:  # circle
            return f"{semicircle_code}\n{semicircle_code}"
            
        
#Special Shapes            
    def generate_greek_spiral(self, size: int, left: bool = True) -> str:
        if not (5 <= size <= 9):
            raise ValueError("A greek-spiral must have at least 5 turns")
        direction = "left" if left else "right"
        return f"for i in range({size+1}):\n    forward(1 * i)\n    {direction}(90.0)"

    def generate_staircase(self, n_times: int, length: int, left: bool = True) -> str:
        direction = "left" if left else "right"
        return f"for i in range({n_times}):\n    forward({length})\n    {direction}(90.0)\n    forward({length})\n    {direction}(90.0)\n    forward(0)\n    {direction}(180.0)"

        
    def generate_zigzag(self, n_times: int, length: int, left: bool = True) -> str:
        direction = "left" if left else "right"
        return f"for i in range({n_times}):\n    forward({length})\n    {direction}(90.0)\n    forward({length})\n    {direction}(270.0)"
    
    # this is how star seems to be generated in the ReGAL paper but it doesn't result in symetric stars (it also fails for stars with 6 points completely) Maybe thats the reason why they only inclued stars with 5 points
    #def generate_star(self, sides: int, left: bool = True) -> str:
    #    if not (5 <= sides <= 9):
    #        raise ValueError("Stars must have between 5 and 9 points")
    #    angle = 360 / sides
    #    direction = "left" if left else "right"
    #    return f"for i in range({sides}):\n    forward(16)\n    {direction}({angle})"
    def generate_star(self, sides: int, left: bool = True) -> str:
        if not (5 <= sides <= 9):
            raise ValueError("Stars must have between 5 and 9 points")
        direction = "left" if left else "right"
        if sides == 6:
            return f"for i in range(3):\n    forward(10)\n    {direction}(120.0)\npenup()\n{direction}(90.0)\nforward(6)\n{direction}(210)\npendown()\nfor i in range(3):\n    forward(10)\n    {direction}(120.0)"
        elif sides == 8:
            angle = ((360.0*3) / sides)
            return f"for i in range(8):\n    forward(10)\n    {direction}({angle})"
        else: #odd-numbers
            angle = (180.0 - (180.0 / sides))
            return f"for i in range({sides}):\n    forward(10)\n    {direction}({angle})"

    def generate_space(self, length: int, angle: float = 0.0, left: bool = True) -> str:
        if not (0.0 <= angle <= 360.0):
            raise ValueError("Angle must be between 0.0 and 360.0")
        if angle == 0:
            return f"penup()\nforward({length})\npendown()"
        else:
            direction = "left" if left else "right"
            return f"penup()\nforward({length})\n{direction}({angle})\npendown()"
            
#Position
    def in_a_row(self, n_times: int, sub_program: str, left: bool = True) -> str:
        #indented_program = re.sub(r"(\n)", r"\1    ", sub_program)
        direction = "left" if left else "right"
        return f"for j in range({n_times}):\n    {sub_program}\n    penup()\n    forward(2)\n    {direction}(0.0)\n\n    pendown()"

    def concentric_semicircle(self, n_times: int, semicircle: bool = True, left: bool = True) -> str:
        EPS_ANGLE = 1.0
        EPS_DIST = 0.03490481287456702
        HALF_INF = 180
        direction = "left" if left else "right"
        semicircle_code = (
            f"for i in range({HALF_INF}):\n        forward({EPS_DIST} * j)\n        {direction}({EPS_ANGLE})"
        )
        if semicircle:
            return f"for j in range({n_times+1}):\n    {semicircle_code}"
        else:
            return f"for j in range({n_times+1}):\n    {semicircle_code}\n    {semicircle_code}"

    def concentric_polygon(self, n_times: int, sides: int, length: int, left: bool = True) -> str:
        angle = 360 / sides
        direction = "left" if left else "right"
        return (
            f"for j in range({n_times+1}):\n    for i in range({sides}):\n        forward({length} * j)\n        {direction}({angle})"
        )


# Combination of shapes and positions
    def shape_sequence(self, shapes: str, left: bool = True) -> str:
        return "\n".join(shapes)

    def sub_program(self, shape_sequence: str, locals: bool = True) -> str:
        return f"embed(\"\"\"{shape_sequence}\"\"\", locals())"

    # Snowflake
    def snowflake(self, sides: int, sub_program: str, left: bool = True) -> str:
        direction = "left" if left else "right"
        return f"for j in range({sides}):\n    {sub_program}\n    forward(0)\n    {direction}({360 / sides})"