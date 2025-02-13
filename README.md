# Step CV-Pipeline: model_pack

During the CV Pipeline Model_Pack stage, the following steps take place:
1. Model conversion     
   The model trained in the previous CV-Pipeline Model_Train stage is converted into a format suitable for specific scenarios. For example, if the REST CV-Pipeline scenario is chosen, the model may be converted into the ONNX format, which enables deploying the model as a REST service.
2. Packaging into bentoservice     
   After model conversion, the model weights and all necessary artifacts (e.g., test image, predictions on the test image) are packaged into bentoservice. Packaging into bentoservice allows creating a containerized application that can be easily deployed and used for inference (prediction) on new data.