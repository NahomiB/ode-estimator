import json


class ODESystemAnalyzer:
    def __init__(self, json_file):
        with open(json_file, "r") as file:
            self.systems = json.load(file)

    def summarize(self):
        total_systems = len(self.systems)

        equations_ranges = [len(system["equations"]) for system in self.systems]
        max_equations = max(equations_ranges)
        min_equations = min(equations_ranges)

        variable_ranges = [len(system["variables"]) for system in self.systems]
        max_variables = max(variable_ranges)
        min_variables = min(variable_ranges)

        parameter_ranges = [len(system["parameters"]) - len(system["restrictions"]) for system in self.systems]
        max_parameters = max(parameter_ranges)
        min_parameters = min(parameter_ranges)

        systems_with_inputs = [
            system for system in self.systems if "inputs" in system and system["inputs"]
        ]
        input_count = len(systems_with_inputs)
        input_ranges = [
            len(system["inputs"]) for system in systems_with_inputs
        ]
        input_range_str = f"{min(input_ranges)} to {max(input_ranges)}" if input_ranges else "N/A"

        systems_with_leaks = [
            system for system in self.systems if any("leak" in param for param in system["parameters"])
        ]
        leak_count = len(systems_with_leaks)
        leak_ranges = [
            len([param for param in system["parameters"] if "leak" in param])
            for system in systems_with_leaks
        ]
        leak_range_str = f"{min(leak_ranges)} to {max(leak_ranges)}" if leak_ranges else "N/A"

        print("=== ODE System Summary ===")
        print(f"Total systems: {total_systems}")
        print(f"Number of equations: {min_equations} to {max_equations}")
        print(f"Number of variables: {min_variables} to {max_variables}")
        print(f"Number of parameters: {min_parameters} to {max_parameters}")
        print(f"Systems with inputs: {input_count}")
        print(f"Input range: {input_range_str}")
        print(f"Systems with leak parameters: {leak_count}")
        print(f"Leak range: {leak_range_str}")



