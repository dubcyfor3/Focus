import pandas as pd

class Buffer:
    def __init__(self, name: str, buffer_models: dict[str, (int, int)], capacity_bytes, bitwidth_bytes):
        self.name = name
        self.buffer_models = buffer_models
        self.models = []
        self.sram_number = []
        self.count = []
        for model, (sram_number, count) in self.buffer_models.items():
            self.sram_number.append(sram_number)
            self.count.append(count)
            self.models.append(BufferModel(model))

        self.capacity_bytes = 0
        self.bitwidth_bytes = 0
        for model, (sram_number, count) in self.buffer_models.items():
            model_height = model.split('x')[0]
            model_width = model.split('x')[1]
            self.capacity_bytes += int(model_width) * int(model_height) // 8 * sram_number * count
            self.bitwidth_bytes += int(model_width) // 8 * sram_number # does not consider count here
        
        assert self.capacity_bytes >= capacity_bytes, f"Capacity bytes mismatch for {self.name}, {self.capacity_bytes} < {capacity_bytes}"
        assert self.bitwidth_bytes >= bitwidth_bytes, f"Bitwidth bytes mismatch for {self.name}, {self.bitwidth_bytes} < {bitwidth_bytes}"
    
    def get_peak_power_mW(self, frequency: int = 500000000) -> float:
        peak_power_mW = 0
        for i in range(len(self.models)):
            peak_power_mW += self.models[i].get_peak_power_mW(frequency) * self.sram_number[i] * self.count[i]
        return peak_power_mW

    def get_area_mm2(self) -> float:
        area_mm2 = 0
        for i in range(len(self.models)):
            area_mm2 += self.models[i].get_area_mm2() * self.sram_number[i] * self.count[i]
        return area_mm2

class BufferModel:
    def __init__(self, model):
        self.model = model
        self.get_spec_from_csv()

    def get_spec_from_csv(self):
        import os
        spec_csv_file = os.path.join(os.path.dirname(__file__), 'buffer_model_spec.csv')
        df = pd.read_csv(spec_csv_file)
        # get the row where capacity_bytes and io_width_bytes are equal to the self.capacity_bytes and self.io_width_bytes, there should be only one row
        row = df.loc[(df['Model'] == self.model)]
        # if no such row, raise an error
        if row.empty:
            raise ValueError(f"No specification found for model: {self.model}")
        self.area_mm2 = row['Area(mm2)'].values[0]
        self.leakage_current_uA = row['Leakage(uA)'].values[0]
        self.read_current_uA_MHz = row['Readc(uA/MHz)'].values[0]
        self.write_current_uA_MHz = row['Writec(uA/MHz)'].values[0]
        self.voltage_V = row['Voltage(V)'].values[0]


    def get_peak_power_mW(self, frequency: int = 500000000) -> float:
        frequency_MHz = frequency / 1000000
        leakage_power_mW = self.leakage_current_uA * self.voltage_V / 1000
        dynamic_power_mW = (self.read_current_uA_MHz + self.write_current_uA_MHz) / 2 * frequency_MHz * self.voltage_V / 1000
        return leakage_power_mW + dynamic_power_mW
    
    def get_area_mm2(self) -> float:
        return self.area_mm2


def get_buffer_stats_from_config_list(name: str, model_list: list[tuple[str, int, int]], capacity: int, bitwidth: int):

    buffer_models = {model: (sram_number, count) for model, sram_number, count in model_list}
    buffer = Buffer(name=name, buffer_models=buffer_models, capacity_bytes=capacity, bitwidth_bytes=bitwidth)
    area = buffer.get_area_mm2()
    peak_power = buffer.get_peak_power_mW()
    return area, peak_power

if __name__ == "__main__":
    buffer_models = {
        '1024x128': (4, 2)
    }
    capcaity = 128 * 1024
    bitwidth = 64
    buffer = Buffer(name="test", buffer_models=buffer_models, capacity_bytes=capcaity, bitwidth_bytes=bitwidth)
    print(buffer.get_peak_power_mW())