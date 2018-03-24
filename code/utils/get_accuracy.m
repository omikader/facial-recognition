function [ accuracy ] = get_accuracy( predictions, testing_data )
%GET_ACCURACY Given the predictions and the testing data, calculate the
%accuracy of the predictions.
%   accuracy = get_accuracy(predictions, testing_data) will return the
%   correctness (between 0 and 1) of a given model's predictions

num_classes = size(testing_data, 3);
num_samples_per_class = size(testing_data, 2);

num_correct = 0;
for i = 1:num_classes
    for n = 1:num_samples_per_class
        if predictions(n, i) == i
            num_correct = num_correct + 1;
        end
    end
end

accuracy = num_correct / (num_classes * num_samples_per_class);