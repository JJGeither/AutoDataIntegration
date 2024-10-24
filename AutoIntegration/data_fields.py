# data_fields.py

class DataField:
    """
    Base class for representing a data field.
    """

    def __init__(self, value):
        self.value = value
        self.field_name = self.__class__.__name__.lower()

    def display(self):
        raise NotImplementedError("Subclasses must implement this method.")


class NameField(DataField):
    def convert(self):
        self.value = [name.split()[0].capitalize() for name in self.value]

    def display(self):
        return f"{self.field_name}: {self.value}"


class DateField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"


class TimeField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"


class TemperatureField(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}°C"

class Etc(DataField):
    def display(self):
        return f"{self.field_name}: {self.value}"


# Additional DataField subclasses follow the same pattern as above
# (StatusField, AddressField, IDField, etc.)


class DataFieldFactory:
    """
    Factory class for creating DataField instances.
    """

    @staticmethod
    def create(field_type, value):
        field_types = {
            "name": NameField,
            "date": DateField,
            "time": TimeField,
            "temperature": TemperatureField,
            "etc": Etc
            # Add other fields as needed
        }

        field_class = field_types.get(field_type.lower())
        if not field_class:
            field_class = field_types.get("etc")

        return field_class(value)
