function area = compute_triangle_areas(X)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    edge = cell(3, 1);

    for j = 1:3
        edge{j} = X.vert(X.triv(:, j), :);
    end

    area = 0.5 .* sqrt(sum(cross(edge{1} - edge{2}, edge{1} - edge{3}).^2, 2));
end
