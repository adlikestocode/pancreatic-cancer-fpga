function results = evaluate_cnn_model()
    try
        load('/content/trained_cnn.mat', 'net');
        load('/content/datastores.mat');
        fprintf('🔍 Evaluating on test set...\n');
        pred = classify(net, imdsTest);
        true_labels = imdsTest.Labels;
        acc = mean(pred == true_labels) * 100;
        [cm, classes] = confusionmat(true_labels, pred);

        fprintf('🏆 Test Accuracy: %.2f%%\n', acc);
        fprintf('Confusion Matrix:\n');
        disp(cm);
        fprintf('Classes: %s\n', mat2str(classes'));

        results = struct('accuracy', acc, 'predictions', pred, ...
                        'confusion_matrix', cm, 'classes', classes);
        save('/content/cnn_results.mat', 'results');
        fprintf('✅ Results saved to cnn_results.mat\n');
    catch ME
        fprintf('❌ Evaluation failed: %s\n', ME.message);
        results = [];
    end
end
