import random
class LOGOProgramSampler:
    def __init__(self, generator_class):
        self.generator = generator_class
        self.generated = set()  # Track generated programs to avoid duplicates

    def _generate_random_program(self):
        """
        Generate a random program based on the grammar dynamically.
        """
        description = ""
        rule = random.choice(["Position", "Shape-Sequence", "SpecialShape", "Snowflake"])
        
        if rule == "Position":
            program, desc = self._generate_random_position()
        elif rule == "Shape-Sequence":
            program, desc = self._generate_random_shape_sequence()
        elif rule == "SpecialShape":
            program, desc = self._generate_random_special_shape()
        elif rule == "Snowflake":
            program, desc = self._generate_random_snowflake()
        
        description += desc
        return program, description

    def _generate_random_position(self):
        """
        Generate a random position program.
        """
        description = ""
        position_type = random.choice(["InARow", "Concentric"])
        
        if position_type == "InARow":
            sub_program, sub_desc = self._generate_random_shape()
            n_times = random.randint(1, 9)
            sub_desc = sub_desc.removeprefix("a ")
            description += f"{n_times} {sub_desc} in a row"
            return self.generator.in_a_row(n_times, sub_program), description
        
        elif position_type == "Concentric":
            shape_type = random.choice(["semicircle", "polygon"])
            n_times = random.randint(2, 9)
            if shape_type == "semicircle":
                description += f"{n_times} concentric circles"
                return self.generator.concentric_semicircle(n_times, semicircle=False), description
            else:
                sides = random.randint(3, 9)
                length = random.choice([2, 4, 20])
                size_desc = "small" if length == 2 else "medium" if length == 4 else "big"
                shape_desc = "triangle" if sides == 3 else "square" if sides == 4 else f"{sides}-gon"
                description += f"{n_times} concentric {size_desc} {shape_desc}"
                return self.generator.concentric_polygon(n_times, sides, length), description

    def _generate_random_shape_sequence(self, snowflake_arm: bool = False):
        """
        Generate a random sequence of shapes.
        """
        if snowflake_arm:
            n_shapes = random.randint(2, 5) # snowflake arms can also start with a space
        else:
            n_shapes = random.randint(3, 5) # normal shape-sequences shouldn't start with a space and to deserve being described to be connected or separated they have to consist of at least 3 shapes
        shapes = []
        descriptions = []
        connected_separated = random.choice([True, False])
        
        for n in range(n_shapes):
            if connected_separated:
                shape, desc = self._generate_random_shape()
            else:
                if not snowflake_arm and n == 0:                # if it is not the arm of a snowflake,
                    shape, desc = self._generate_random_shape() # the first element is a shape
                elif n == n_shapes - 1:
                    shape, desc = self._generate_random_shape() # and the last element is always a shape
                else:
                    if any("space" in d for d in descriptions) or random.choice([True, False]):
                        shape, desc = self._generate_random_shape()
                    else:
                        length = random.choice([2, 4, 20])
                        angle = random.choice([0, 90, 180, 270])
                        shape = self.generator.generate_space(length, angle)
                        size_desc = "short" if length == 2 else "medium" if length == 4 else "big"
                        desc = f"a {size_desc} space"
            shapes.append(shape)
            descriptions.append(desc)
        
        description = f"{'connected' if connected_separated else 'separated'} sequence of shapes: " + ", ".join(descriptions)
        return self.generator.shape_sequence(shapes), description

    def _generate_random_shape(self):
        """
        Generate a random shape.
        """
        options = ["Line", "Polygon", "Semicircle"]
        weights = [1/10, 7/10, 2/10]
        shape_type = random.choices(population=options, weights=weights, k=1)[0]
        description = ""

        if shape_type == "Line":
            length = random.choice([2, 4, 20])
            angle = random.choice([0, 90, 180, 270])
            size_desc = "short" if length == 2 else "medium" if length == 4 else "big"
            description += f"a {size_desc} line"
            return self.generator.generate_line(length, angle), description
        
        elif shape_type == "Polygon":
            sides = random.randint(3, 9)
            length = random.choice([2, 4, 20])
            size_desc = "small" if length == 2 else "medium" if length == 4 else "big"
            shape_desc = "triangle" if sides == 3 else "square" if sides == 4 else f"{sides}-gon"
            description += f"a {size_desc} {shape_desc}"
            return self.generator.generate_polygon(sides, length), description
        
        elif shape_type == "Semicircle":
            semicircle = random.choice([True, False])
            size = random.choice([1, 2])
            size_desc = "small" if size == 1 else "medium"
            shape_desc = "semicircle" if semicircle else "circle"
            description += f"a {size_desc} {shape_desc}"
            return self.generator.generate_semicircle(size, semicircle=semicircle), description
    
    def _generate_random_special_shape(self):
        """
        Generate a random special shape.
        """
        special_shape_type = random.choice(["GreekSpiral", "Staircase", "Zigzag", "Star"])
        description = ""
        
        if special_shape_type == "GreekSpiral":
            size = random.randint(5, 9)
            description += f"a greek spiral with {size} turns"
            return self.generator.generate_greek_spiral(size), description
        elif special_shape_type == "Staircase":
            n_times = random.randint(2, 9)
            length = random.choice([2, 4, 20])
            size_desc = "small" if length == 2 else "medium" if length == 4 else "big"
            description += f"a staircase with {n_times} {size_desc} steps"
            return self.generator.generate_staircase(n_times, length), description
        elif special_shape_type == "Zigzag":
            n_times = random.randint(2, 9)
            length = random.choice([2, 4, 20])
            size_desc = "small" if length == 2 else "medium" if length == 4 else "big"
            description += f"a zigzag with {n_times} {size_desc} steps"
            return self.generator.generate_zigzag(n_times, length), description
        elif special_shape_type == "Star":
            sides = random.randint(5, 9)
            description += f"a star with {sides} points"
            return self.generator.generate_star(sides), description

    def _generate_random_snowflake(self):
        """
        Generate a random snowflake program.
        """
        sides = random.randint(3, 8)
        n_arms = random.randint(1, 2)
        description = f"a {sides} sided snowflake with "

        if n_arms == 1:
            arm , arm_description = self._generate_random_shape()   # contains a single shape
            description += f"an arm of {arm_description.lower()}"

        else:
            arm, arm_description = self._generate_random_shape_sequence(snowflake_arm=True) # contains multiple shapes
            description += f"arms of {arm_description.lower()}"

        sub_program = self.generator.sub_program(arm)
        return self.generator.snowflake(sides, sub_program), description

    def sample(self, n=1):
        """
        Dynamically generate `n` unique random programs.
        """
        samples = []
        while len(samples) < n:
            program, description = self._generate_random_program()
            if program not in self.generated:
                self.generated.add(program)
                samples.append({"Program": program, "Description": description})
        return samples

    def reset(self):
        """
        Reset the sampler to allow generating duplicates.
        """
        self.generated = set()