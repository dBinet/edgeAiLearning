"""
Week 18 Session 4 — TFLite quantization scope (Conv2d vs Linear)

Goal: determine whether TFLite's INT8 post-training quantization quantizes
Conv2d layers or only Linear layers, by comparing tensor dtypes between the
FP32 and INT8 TFLite models from Sessions 1-2. Same question Week 17
Session 3 answered for ONNX (via ConvInteger ops) and PyTorch (Linear/LSTM
only) - same investigation, new framework.

Note: the INT8 model has 2 more tensors than FP32 (input_quantized_internal,
output_dequantized_output) - quantized models need explicit quantize/
dequantize wrapper tensors at the float32/int8 boundary. Matching tensors
by name (not list position) sidesteps this entirely.
"""

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


TFLITE_FP32_PATH = "models/tflite_week18/cnn_week16_session3_float32.tflite"
TFLITE_INT8_PATH = "models/tflite_week18/cnn_week16_session3_integer_quant.tflite"


def load_tflite_interpreter(tflite_path):
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter


if __name__ == "__main__":
    fp32_interp = load_tflite_interpreter(TFLITE_FP32_PATH)
    int8_interp = load_tflite_interpreter(TFLITE_INT8_PATH)

    fp32_by_name = {t['name']: t for t in fp32_interp.get_tensor_details()}
    int8_details = int8_interp.get_tensor_details()

    print(f"{'Tensor':35s} {'FP32 dtype':15s} {'INT8 dtype':15s}")
    print("-" * 65)

    conv_quantized = []
    linear_quantized = []

    for int8_t in int8_details:
        name = int8_t['name']
        if "Conv_conv_filter" in name or "Gemm_fc_weights" in name:
            fp32_t = fp32_by_name.get(name)
            if fp32_t is None:
                print(f"{name:35s} (not found in FP32 model)")
                continue

            fp32_dtype = fp32_t['dtype'].__name__
            int8_dtype = int8_t['dtype'].__name__
            print(f"{name:35s} {fp32_dtype:15s} {int8_dtype:15s}")

            if "Conv_conv_filter" in name:
                conv_quantized.append(int8_dtype == "int8")
            else:
                linear_quantized.append(int8_dtype == "int8")

    print()
    if all(conv_quantized) and all(linear_quantized):
        print("Conclusion: TFLite INT8 quantizes Conv2d AND Linear weights "
              "to int8 - matches ONNX Runtime's scope, not PyTorch's "
              "Linear/LSTM-only dynamic quantization.")
    else:
        print(f"Conclusion: Conv2d quantized: {all(conv_quantized)}, "
              f"Linear quantized: {all(linear_quantized)}")