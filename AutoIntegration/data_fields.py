import re
from dateutil import parser
EMPTYVALUE = 'NULL'

class DataField:
    """
    Base class for representing a data field.
    """

    def __init__(self, value):
        self.value = value
        self.field_name = self.__class__.__name__.lower()

    def display(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def convert(self):
        print("No Conversion")
        return 0


class NameField(DataField):
    def convert(self):
        if isinstance(self.value[0], str):
            converted_names = [self.value[0]]
            for name in self.value[1:]:
                if isinstance(name, str) and name != EMPTYVALUE:
                    # Convert to lowercase and strip any extra whitespace
                    name = name.strip().lower()

                    # Remove all periods
                    name = name.replace('.', '')

                    # Handle "last, first" format
                    if ',' in name:
                        parts = [part.strip() for part in name.split(',')]
                        name = f"{parts[1]} {parts[0]}"

                    # Remove middle initials or extra spaces
                    name = re.sub(r'\b[a-zA-Z]\.?\b', '', name)  # Remove middle name
                    name = ' '.join(name.split())

                    # Capitalize first and last names
                    name = ' '.join(part.capitalize() for part in name.split())
                    converted_names.append(name)
                else:
                    converted_names.append(EMPTYVALUE)
            self.value = converted_names


class DateField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"

    def convert(self):
        if all(isinstance(date, str) for date in self.value[1:]):
            standardized_dates = [self.value[0]]  # Keep the header or label unchanged
            for date in self.value[1:]:
                try:
                    parsed_date = parser.parse(date.strip())
                    # Format the date to YYYY-MM-DD
                    standardized_date = parsed_date.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    standardized_date = "Invalid Date"
                standardized_dates.append(standardized_date)

            self.value = standardized_dates


class TimeField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"

    def convert(self):
        if all(isinstance(time, str) for time in self.value[1:]):
            converted_times = [self.value[0]]
            for time in self.value[1:]:
                if isinstance(time, str) and time != EMPTYVALUE:
                    time = time.strip().lower()

                    # Check if the time is AM or PM (or shorthand 'a' or 'p')
                    am_pm = None
                    if 'a' in time:
                        am_pm = 'AM'
                        time = time.replace('a', '').strip()
                    elif 'p' in time:
                        am_pm = 'PM'
                        time = time.replace('p', '').strip()

                    time = re.sub(r'[^\d:]', '', time)  # Remove non-numeric characters except ':'

                    # Split the time into components (HH, MM, SS)
                    time_parts = time.split(':')

                    # Ensure that there are exactly three components (HH, MM, SS)
                    if len(time_parts) == 2:
                        time_parts.insert(2, '00')  # Assume 00 for seconds unless provided
                    elif len(time_parts) == 1:
                        time_parts = [time_parts[0], '00', '00']  # Assume 00 for seconds and minutes unless provided

                    # Ensure there are exactly 3 components and all are numeric
                    time_parts = [str(part).zfill(2) for part in time_parts[:3]]

                    hour = int(time_parts[0])
                    if am_pm == 'AM':
                        if hour == 12:
                            time_parts[0] = '00'

                    elif am_pm == 'PM':
                        if hour != 12:
                            time_parts[0] = str(hour + 12)

                    formatted_time = ':'.join(time_parts)

                    converted_times.append(formatted_time)
                else:
                    converted_times.append(EMPTYVALUE)

            self.value = converted_times


class TemperatureField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}°C"

    def convert(self):
        if all(isinstance(val, str) for val in self.value[1:]):
            is_celsius = None

            # Check if any of the values explicitly mention 'C' or 'F'
            for val in self.value[1:]:
                if 'C' in val.upper():
                    is_celsius = True
                    break
                elif 'F' in val.upper():
                    is_celsius = False
                    break

            # Calculate the average temperature and decide
            if is_celsius is None:
                numeric_values = [float(re.sub(r'[^\d.-]', '', val)) for val in self.value[1:]]
                avg_temp = sum(numeric_values) / len(numeric_values)

                # If the average is near freezing point (0°C) or below, it's likely Celsius
                if avg_temp < 50:
                    is_celsius = True
                else:
                    is_celsius = False

            # The label
            converted_temperatures = [self.value[0]]

            for val in self.value[1:]:
                # Strip any extra spaces and handle values
                temp = val.strip().upper()
                if temp != EMPTYVALUE:
                    if 'F' in temp:  # Fahrenheit value detected
                        temp_value = float(re.sub(r'[^\d.-]', '', temp))
                        if is_celsius:
                            # Convert Fahrenheit to Celsius
                            temp_value = (temp_value - 32) * 5 / 9
                    else:  # No unit detected, assume it's Celsius by default
                        temp_value = float(re.sub(r'[^\d.-]', '', temp))
                        if not is_celsius:
                            # Convert Celsius to Fahrenheit
                            temp_value = temp_value * 9 / 5 + 32

                    # Format the temperature to 1 decimal place for readability
                    converted_temperatures.append(f"{temp_value:.1f}")
                else:
                    converted_temperatures.append(EMPTYVALUE)

            # Update the values with the converted temperatures
            self.value = converted_temperatures

class Number(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}°C"

    def convert(self):
        # Remove all commas from each value
        converted_numbers = [self.value[0]]
        for num in self.value[1:]:
            if num != EMPTYVALUE:
                num = num.replace(',', '')  # Remove commas
                converted_numbers.append(num)
            else:
                converted_numbers.append(EMPTYVALUE)

        self.value = converted_numbers


class Etc(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"


def get_categorical_subclasses():
    field_types = {
        "name": [NameField, "John A. Smith"],  # Example sentence for a name field
        "date": [DataField, "date 5/12/2023 3/23/98 January 25,2024"],  # Example sentence for a date field
        "time": [TimeField, "time 10:30 AM"],  # Example sentence for a time field
        "temperature": [TemperatureField, "22.5 degrees Celsius"],  # Example sentence for a temperature field
        "address": [Etc, "123 Main Street, Springfield, IL"],  # Example sentence for an address field
        "money": [Number, "the salary is $10 and $13.23 and 6.23"],
        "etc": [Etc, "miscellaneous data"]  # Example sentence for an etc field
        # Add other fields as needed
    }
    return field_types


class DataFieldFactory:
    """
    Factory class for creating DataField instances.
    """
    @staticmethod
    def create(field_type, value):
        field_types = get_categorical_subclasses()

        field_class = field_types.get(field_type.lower())
        field_class = field_class[0]
        if not field_class:
            field_class = field_types.get("etc")

        return field_class(value)
