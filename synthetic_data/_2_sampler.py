import random
class LOGOProgramSampler:
    def __init__(self, generator_class, *dfs):
        self.generator = generator_class
        self.generated = set()  # Track generated programs to avoid duplicates

        for df in dfs:
            if df is not None and "Program" in df.columns:
                self.generated.update(df["Program"].tolist())


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
            shape, sub_desc = self._generate_random_shape()
            sub_program = self.generator.sub_program(shape)
            n_times = random.randint(2, 9)
            sub_desc = sub_desc.removeprefix("a ")
            description += f"{n_times} {sub_desc} in a row"
            return self.generator.in_a_row(n_times, sub_program), description
        
        elif position_type == "Concentric":
            shape_type = random.choice(["semicircle", "polygon"])
            n_times = random.randint(3, 9)
            if shape_type == "semicircle":
                description += f"{n_times} concentric circles"
                return self.generator.concentric_semicircle(n_times, semicircle=False), description
            else:
                sides = random.randint(3, 9)
                length = random.choice([2, 4])
                size_desc = "small" if length == 2 else "medium" 
                shape_desc = "triangle" if sides == 3 else "square" if sides == 4 else f"{sides}-gon"
                description += f"{n_times} concentric {size_desc} {shape_desc}"
                return self.generator.concentric_polygon(n_times, sides, length), description

    def _generate_random_shape_sequence(self, snowflake_arm: bool = False):
        """
        Generate a random sequence of shapes.
        """
        if snowflake_arm:
            n_shapes = random.randint(2, 3) # snowflake arms should consist of at most 3 shapes
            even_odd = random.choice([True, False]) 
        else:
            n_shapes = random.randint(3, 5) 
            even_odd = True
        shapes = []
        descriptions = []
        connected_separated = random.choice([True, False])

        for n in range(n_shapes):
            if even_odd:
                if n % 2 == 0:
                    shape, desc  = self._generate_random_shape(options=["Polygon", "Semicircle"], weights=[7/9, 2/9])
                else:
                    if connected_separated:
                        shape, desc  = self._generate_random_shape(options=["Line"], weights=[1])
                    else:
                        length = random.choice([2, 4, 20])
                        angle = random.choice([0, 90, 180, 270])
                        shape = self.generator.generate_space(length, angle)
                        size_desc = "short" if length == 2 else "medium" if length == 4 else "big"
                        desc = f"a {size_desc} space"
            else:
                if n % 2 == 0:
                    if connected_separated:
                        shape, desc  = self._generate_random_shape(options=["Line"], weights=[1])
                    else:
                        length = random.choice([2, 4, 20])
                        angle = random.choice([0, 90, 180, 270])
                        shape = self.generator.generate_space(length, angle)
                        size_desc = "short" if length == 2 else "medium" if length == 4 else "big"
                        desc = f"a {size_desc} space"
                else:
                    shape, desc  = self._generate_random_shape(options=["Polygon", "Semicircle"], weights=[7/9, 2/9])
            shapes.append(shape)
            descriptions.append(desc)
        
        description = f"{'connected' if connected_separated else 'separated'} sequence of shapes: " + ", ".join(descriptions)
        return self.generator.shape_sequence(shapes), description

    def _generate_random_shape(self, options=["Line", "Polygon", "Semicircle"], weights=[1/10, 7/10, 2/10]):
        """
        Generate a random shape.
        """
        options = options
        weights = weights
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
            length = random.choice([2, 4])
            size_desc = "small" if length == 2 else "medium" 
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
            length = random.choice([2, 4])
            size_desc = "small" if length == 2 else "medium" 
            description += f"a staircase with {n_times} {size_desc} steps"
            return self.generator.generate_staircase(n_times, length), description
        elif special_shape_type == "Zigzag":
            n_times = random.randint(2, 9)
            length = random.choice([2, 4])
            size_desc = "small" if length == 2 else "medium" 
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

    def count_possible_shapes(self):
        """
        Analyze the theoretical number of possible unique shapes that can be generated.
        Returns a dictionary with counts for each shape category and total.
        """
        # Basic shapes
        line_count = 3 * 4  # 3 sizes × 4 angles = 12
        polygon_count = 7 * 2  # 7 types (3-9 sides) × 2 sizes = 14
        semicircle_count = 2 * 2  # 2 types (semicircle, circle) × 2 sizes = 4
        basic_shapes_count = line_count + polygon_count + semicircle_count  # 30
        
        # For sequence calculations
        polygon_semicircle_count = polygon_count + semicircle_count  # 18
        
        # Special shapes
        greek_spiral_count = 5  # 5-9 turns
        staircase_count = 8 * 2  # 2-9 steps × 2 sizes = 16
        zigzag_count = 8 * 2  # 2-9 steps × 2 sizes = 16
        star_count = 5  # 5-9 points
        special_shapes_count = greek_spiral_count + staircase_count + zigzag_count + star_count  # 42
        
        # Sequences - calculate exact count based on rules
        # Rule: n_shapes = 3-5 for normal sequences, 2-3 for snowflake arms
        # Even-odd pattern with alternating shape types
        # Connected vs separated option
        
        # Normal sequences (3-5 shapes)
        # For each length, we calculate even_odd=True and even_odd=False cases
        # For each case, we calculate connected and separated variants
        normal_sequence_count = 0
        
        for length in range(3, 6):  # 3, 4, 5 shapes
            # Even-odd pattern = True (starts with polygon/semicircle)
            # For n positions where n % 2 == 0: polygon_semicircle_count options
            # For n positions where n % 2 == 1:
            #    - Connected: line_count options
            #    - Separated: space_count options (3 sizes × 4 angles = 12)
            
            polygon_positions = (length + 1) // 2  # Ceiling division for odd lengths
            line_positions = length // 2  # Floor division for even lengths
            
            # Connected sequence
            connected_count = (polygon_semicircle_count ** polygon_positions) * (line_count ** line_positions)
            
            # Separated sequence
            space_count = 3 * 4  # 3 sizes × 4 angles for spaces = 12 (same as line_count)
            separated_count = (polygon_semicircle_count ** polygon_positions) * (space_count ** line_positions)
            
            # Even-odd pattern = False (starts with line/space)
            # For n positions where n % 2 == 1: polygon_semicircle_count options
            # For n positions where n % 2 == 0:
            #    - Connected: line_count options
            #    - Separated: space_count options
            
            polygon_positions_alt = length // 2  # Now polygons are in odd positions
            line_positions_alt = (length + 1) // 2  # Now lines/spaces are in even positions
            
            # Connected sequence (pattern flipped)
            connected_count_alt = (polygon_semicircle_count ** polygon_positions_alt) * (line_count ** line_positions_alt)
            
            # Separated sequence (pattern flipped)
            separated_count_alt = (polygon_semicircle_count ** polygon_positions_alt) * (space_count ** line_positions_alt)
            
            normal_sequence_count += connected_count + separated_count + connected_count_alt + separated_count_alt
        
        # Snowflake arm sequences (2-3 shapes)
        snowflake_arm_sequence_count = 0
        
        for length in range(2, 4):  # 2-3 shapes
            # Even-odd can be True or False for snowflake arms
            # Same calculation logic as normal sequences
            
            # Even-odd pattern = True
            polygon_positions = (length + 1) // 2
            line_positions = length // 2
            
            connected_count = (polygon_semicircle_count ** polygon_positions) * (line_count ** line_positions)
            space_count = 3 * 4  # Same as before
            separated_count = (polygon_semicircle_count ** polygon_positions) * (space_count ** line_positions)
            
            # Even-odd pattern = False
            polygon_positions_alt = length // 2
            line_positions_alt = (length + 1) // 2
            
            connected_count_alt = (polygon_semicircle_count ** polygon_positions_alt) * (line_count ** line_positions_alt)
            separated_count_alt = (polygon_semicircle_count ** polygon_positions_alt) * (space_count ** line_positions_alt)
            
            snowflake_arm_sequence_count += connected_count + separated_count + connected_count_alt + separated_count_alt
        
        # Snowflakes
        snowflake_sides = 6  # 3-8 sides
        
        # Snowflakes with single shape arm
        snowflake_single_arm_count = snowflake_sides * basic_shapes_count
        
        # Snowflakes with sequence arm - now using our exact calculation
        snowflake_sequence_arm_count = snowflake_sides * snowflake_arm_sequence_count
        
        # Position arrangements
        in_a_row_count = 8 * basic_shapes_count  # 2-9 repetitions × basic shapes
        concentric_count_semicircle = 7  # 3-9 concentric circles
        concentric_count_polygon = 7 * 7 * 2  # 3-9 concentric × 7 polygon types × 2 sizes
        position_count = in_a_row_count + concentric_count_semicircle + concentric_count_polygon
        
        # Total sequence count is normal sequences + those used in snowflake arms
        sequence_count = normal_sequence_count + snowflake_arm_sequence_count
        
        # Totals
        total_count = basic_shapes_count + special_shapes_count + normal_sequence_count + snowflake_single_arm_count + snowflake_sequence_arm_count + position_count
        
        # Calculate relative frequencies
        total_as_float = float(total_count)
        rel_basic = basic_shapes_count / total_as_float * 100
        rel_special = special_shapes_count / total_as_float * 100
        rel_sequence = normal_sequence_count / total_as_float * 100
        rel_snowflake_single = snowflake_single_arm_count / total_as_float * 100
        rel_snowflake_seq = snowflake_sequence_arm_count / total_as_float * 100
        rel_position = position_count / total_as_float * 100
        
        # Include detailed counts of each shape type
        detail = {
            "Lines": line_count,
            "Polygons": polygon_count,
            "Semicircles": semicircle_count,
            "Greek Spirals": greek_spiral_count,
            "Staircases": staircase_count,
            "Zigzags": zigzag_count,
            "Stars": star_count,
            "Normal Sequences": normal_sequence_count,
            "Snowflake Arm Sequences": snowflake_arm_sequence_count,
            "In A Row Arrangements": in_a_row_count,
            "Concentric Arrangements": concentric_count_semicircle + concentric_count_polygon
        }
        
        return {
            "Basic Shapes": {"count": basic_shapes_count, "percent": f"{rel_basic:.2f}%"},
            "Special Shapes": {"count": special_shapes_count, "percent": f"{rel_special:.2f}%"},
            "Sequences": {"count": normal_sequence_count, "percent": f"{rel_sequence:.2f}%"},
            "Snowflakes with Single Shape Arms": {"count": snowflake_single_arm_count, "percent": f"{rel_snowflake_single:.2f}%"},
            "Snowflakes with Sequence Arms": {"count": snowflake_sequence_arm_count, "percent": f"{rel_snowflake_seq:.2f}%"},
            "Position Arrangements": {"count": position_count, "percent": f"{rel_position:.2f}%"},
            "Total": total_count
        }


    def reset(self):
        """
        Reset the sampler to allow generating duplicates.
        """
        self.generated = set()