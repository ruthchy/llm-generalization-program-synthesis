import re

class SemanticLength:
    def __init__(self):
        pass

    def calc_semantic_length(self, program):
        output = []
        stack = [] # Stack to manage nested counts
        
        # Regex patterns for the entire input
        for_loop_pattern = r"for\s+\w+\s+in\s+range\((\d+|HALF_INF)\):" # add the option of HALF_INF which is used to draw simicircles and circles
        embed_start_pattern = r'embed\("""'
        forward_left_pattern = r"(forward|left|right|penup|pendown|teleport|heading|isdown)\(([\w\.\*\+\-\(\)0-9]*)\)"
        embed_end_pattern = r'""", locals\(\)\)'
        
        # Track position in input
        position = 0
        while position < len( program):
            # Match for-loop
            for_match = re.match(for_loop_pattern,  program[position:])
            if for_match:
                loop_value = for_match.group(1)  # Extract the loop value (either a digit or HALF_INF)
                if loop_value == "HALF_INF":  # If the loop is using HALF_INF
                    output.append(f"180*(")  # Append 180 for HALF_INF
                else:
                    n_iter = int(loop_value)  # Convert the loop value to an integer if it's a digit
                    output.append(f"{n_iter}*(")
                stack.append(0)  # track for-loop
                position += for_match.end()
                continue
            
            # Match embed start
            embed_start_match = re.match(embed_start_pattern,  program[position:])
            if embed_start_match:
                output.append("(")
                stack.append(1)  # track embed block
                position += embed_start_match.end()
                continue
            
            # Match forward/left
            forward_left_match = re.match(forward_left_pattern,  program[position:])
            if forward_left_match:
                # Check if the last character in output is a closing bracket
                if output and re.search(r"\)+$", output[-1]):
                    output.append("+1+")  # Append "+1+" if there are closing brackets
                else:
                    output.append("1+")  # Otherwise, append "1+"
                #if stack:
                #    stack[-1] += 1  # Increment the counter for the current block
                position += forward_left_match.end()
                continue
            
            # Match embed end
            embed_end_match = re.match(embed_end_pattern,  program[position:])
            count_zeros = 0
            if embed_end_match:
                if stack:
                    for i in reversed(range(len(stack))):
                        if stack[i] == 1:
                            count_zeros +=1
                            break
                        elif stack[i] == 0:
                            count_zeros +=1
                    stack = stack[:i] # remove all the last elements since starting the embed block
                    if output and output[-1] == "1+":
                        output[-1] = "1"
                    output.append(")" * count_zeros)

                    #count = 0 # set the count to 0 after adding brackets
                position += embed_end_match.end()
                continue
            
            # Increment position if no match is found
            position += 1
        
        # Close any remaining loops
        while stack:
            count = int(len(stack))
            stack = [] # clear the stack
            if output and output[-1] == "1+":
                output[-1] = "1"
            output.append(")"* count) 
    
        # Combine the output
        result = ''.join(output).rstrip('+')
        #print(f"Formular: {result}") # If you want to see the Formular underlying the Semantic Calculation
        try:
            result = eval(result)
        except Exception as e:
            print(f"Error while evaluation the expression: {e}")
        
        return result