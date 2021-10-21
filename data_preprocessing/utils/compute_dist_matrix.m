function D = compute_dist_matrix(S, samples)

    % Copyright (c) Facebook, Inc. and its affiliates.
    %
    % This source code is licensed under the MIT license found in the
    % LICENSE file in the root directory of this source tree.

    M = S.X;
    M.n = size(M.vert, 1);
    M.m = size(M.triv, 1);

    if nargin == 1 || isempty(samples)
        samples = 1:M.n;
    end

    if ~exist('fastmarchmex')
        % Use precomputed binaries from https://github.com/abbasloo/dnnAuto/
        base_url = "https://github.com/abbasloo/dnnAuto/raw/37ce4320bc90a75b07a7ec1d862484d6576cec4c/preprocessing/isc";
        for ext = {'mexa64', 'mexmaci64', 'mexw32', 'mexw64'}
            urlwrite(...
                base_url + "/fastmarchmex." + ext, ...
                "fastmarchmex." + ext);
        end
        rehash
    end

    % Calls legacy fast marching code
    march = fastmarchmex('init', int32(M.triv - 1), double(M.vert(:, 1)), double(M.vert(:, 2)), double(M.vert(:, 3)));

    D = zeros(length(samples));

    for i = 1:length(samples)
        source = inf(M.n, 1);
        source(samples(i)) = 0;
        d = fastmarchmex('march', march, double(source));
        D(:, i) = d(samples);
    end

    fastmarchmex('deinit', march);

    % Ensures that the distance matrix is exactly symmetric
    D = 0.5 * (D + D');
end
