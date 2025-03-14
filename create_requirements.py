'''
First activate the desired conda env and then execute: conda env export > environment.yml
This will create a file called environment.yml in the current directory.
Then run this script to extract the pip dependencies from the environment.yml file and write them to a requirements.txt file.
'''
import yaml

# Load the environment.yml file
with open('environment.yml', 'r') as file:
    env = yaml.safe_load(file)

# Extract pip dependencies
pip_deps = env['dependencies'][-1]['pip']

# Write pip dependencies to requirements.txt
with open('requirements.txt', 'w') as file:
    for dep in pip_deps:
        file.write(f"{dep}\n")

print("requirements.txt file has been created.")