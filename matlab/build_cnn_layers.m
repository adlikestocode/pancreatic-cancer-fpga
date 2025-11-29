function layers = build_cnn_layers()
    layers = [
        imageInputLayer([224 224 3],'Name','input')
        convolution2dLayer(3,16,'Padding','same','Name','conv1')
        reluLayer('Name','relu1')
        maxPooling2dLayer(2,'Stride',2,'Name','pool1')
        convolution2dLayer(3,32,'Padding','same','Name','conv2')
        reluLayer('Name','relu2')
        globalAveragePooling2dLayer('Name','gapool')
        fullyConnectedLayer(3,'Name','fc')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')
    ];

    % Save for HDL Coder compatibility
    save('/content/cnn_layers.mat', 'layers');
    fprintf('✅ CNN layers saved: %d layers\n', numel(layers));
    fprintf('✅ HDL-ready architecture complete\n');
end
