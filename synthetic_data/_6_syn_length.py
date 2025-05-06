import re

class SynLength:
    def __init__(self):
        pass

    def calc_length(self, program):
        length = 0

        # Regex patterns
        for_loop_pattern = r"for\s+\w+\s+in\s+range\((\d+|HALF_INF)\):"
        embed_start_pattern = r'embed\("""'
        embed_end_pattern = r'""",\s*locals\(\)\)'
        command_pattern = r"(forward|left|right|penup|pendown|teleport|heading|isdown)\(([\w\.\*\+\-\(\)0-9]*)\)"

        # Count components
        length += len(re.findall(for_loop_pattern, program))         # Each for-loop = 1
        length += len(re.findall(embed_start_pattern, program))      # Each embed = 1
        length += len(re.findall(embed_end_pattern, program))        # Each locals = 1
        length += len(re.findall(command_pattern, program))          # Each command = 1

        return length