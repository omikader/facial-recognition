function [ data ] = to_vector( dataset, input )
%TO_VECTOR Reshapes images from matrices to feature vectors.
%   data = TO_VECTOR(dataset, input) reshapes the input data from the
%   specified dataset into feature vectors.

switch dataset
    case 'face'
        data = zeros(size(input, 1) * size(input, 2), 3, 200);
        
        for n = 1:200
            data(:, 1, n) = reshape(input(:, :, 3*n-2), [504, 1]);
            data(:, 2, n) = reshape(input(:, :, 3*n-1), [504, 1]);
            data(:, 3, n) = reshape(input(:, :, 3*n), [504, 1]);
        end
    case 'pose'
        data = zeros(size(input, 1) * size(input, 2), 13, 68);

        for n = 1:size(input, 3)
            for i = 1:size(input, 4)
                data(:, n, i) = reshape(input(:, :, n, i), [1920, 1]);
            end
        end
end
end

