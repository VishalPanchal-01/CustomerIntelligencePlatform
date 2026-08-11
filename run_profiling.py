from src.validation.data_profiling import DataProfiller


profiler = DataProfiller()

df = profiler.profile_structure()

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(
    profiler.profile_missing_value(df)
)
print("\nDuplicate Report:")
print(profiler.profile_duplicate(df))

print("\nNumerical report:")
print(
    profiler.profile_numeric_columns(df)
)

print("\nCancellation report:")
print(
    profiler.profile_cancellation(df)
)

print("\nDate report:")
print(
    profiler.profile_dates(df)
)

print("\nBusiness entity report:")
print(
    profiler.profile_business_entities(df)
)

