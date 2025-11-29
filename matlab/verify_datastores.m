function status = verify_datastores()
    try
        load('/content/datastores.mat');
        fprintf('✅ Train: %d files OK\n', numel(imdsTrainAug.Files));
        fprintf('✅ Val: %d files OK\n', numel(imdsVal.Files));
        fprintf('✅ Test: %d files OK\n', numel(imdsTest.Files));

        % Verify first files exist on disk
        if exist(imdsTrainAug.Files{1}, 'file') && ...
           exist(imdsVal.Files{1}, 'file') && ...
           exist(imdsTest.Files{1}, 'file')
            fprintf('✅ FIRST FILES ACCESSIBLE\n');
            status = true;
        else
            fprintf('❌ FILES MISSING\n');
            status = false;
        end
    catch ME
        fprintf('❌ LOAD FAILED: %s\n', ME.message);
        status = false;
    end
end
