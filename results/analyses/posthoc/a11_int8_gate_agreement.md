post hoc, not pre-specified

# A11: int8 vs fp32 gate decisions @25%

not available: no int8 gate scores for 2016 are stored. The int8 gate ONNX files in checkpoints/footprint/ were exported from the validation-time gates (checkpoints/gates/), not from the 2015-refit gates used on 2016 (checkpoints/gates_test/), and were never run on 2016.
