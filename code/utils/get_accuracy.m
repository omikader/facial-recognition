function [ accuracy ] = get_accuracy(predictions, testing_data)
%GET_ACCURACY Given the predictions and the testing data, calculate the
%accuracy of the predictions.
%   accuracy = GET_ACCURACY(predictions, testing_data) will return the
%   correctness (between 0 and 1) of a given model's predictions.

num_classes = size(testing_data, 3);
num_samples_per_class = size(testing_data, 2);

num_correct = 0;
num_labeled = 0;

for i = 1:num_classes
    for n = 1:num_samples_per_class
        guess = predictions(n, i);
        if guess ~= -1
            num_labeled = num_labeled + 1;
            if guess == i
                num_correct = num_correct + 1;
            end
        end
    end
end

accuracy = num_correct / num_labeled;