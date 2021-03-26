function [t, Y, X, Z] = SimulateMarkedHawkes1D(T_stop, mu, Y0, delta, distY, parY, distX, parX, seed)
% Simulate Univariate Hawkes Process
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% REFERENCE: Simulation and Calibration of a Fully Bayesian Marked
% Multidimensional Hawkes Process with Dissimilar Decays, Kar Wai Lim et.
% al.
% CODE AVAILABLE AT: https://karwailim.github.io/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Input:
%        T_stop = maturity time
%            mu = background intensity (scalar)
%            Y0 = intensity from edge effect (scalar)
%         delta = decay rates (scalar)
%         distY = distribution of Y (intensity jump)
%          parY = parameters for Y (1 * 1 * NY tensor)
%
%  Optional Input:
%         distX = distribution of X (marked)
%          parX = parameters for X (1 * 1 * NX tensor)
%          seed = seed for random number generator
%
%  Output:
%             t = event times (Nm * 1 vector)
%             X = marked      (Nm * 1 vector)
%             Y = jump sizes  (Nm * 1 vector) 
%             Z = origins     (Nm * 1 vector)
%
%
%  Note: 
%       1) Supported distY: 'const', 'gamma', 'exponential', 'iidGBM', 'corrLogNorm'
%       2) Supported distX: 'const', 'gamma', 'exponential', 'normal'
%       3) parX has additional column for marked from immigrant
%

if exist('seed','var')
    disp('seed exists')
    rng(seed);
else
    seed = int32((now - 7.361e5) * 1e6);
    disp(['generated seed: ' num2str(seed)]);
    rng(seed);
end


%% Parameter Setting

N = 10000000; % pre-allocate array size


%% Cache Variables

inv_mu    = 1 ./ mu;
inv_delta = 1 ./ delta;

% boolean on the sampling dist. of Y
Yconst = strcmp(distY, 'const');
Ygam   = strcmp(distY, 'gamma');
Yexp   = strcmp(distY, 'exponential');
Ygbm   = strcmp(distY, 'iidGBM');
Ycln   = strcmp(distY, 'corrLogNorm');

% boolean on the sampling dist. of X
if exist('distX','var')
    Xconst = strcmp(distX, 'const');
    Xgam   = strcmp(distX, 'gamma');
    Xexp   = strcmp(distX, 'exponential');
    Xnorm  = strcmp(distX, 'normal');
else
    Xconst = false;
    Xgam   = false;
    Xexp   = false;
    Xnorm  = false;
end

% boolean on whether to store variables
saveZ = nargout >= 4;
saveX = nargout >= 3;
saveY = (nargout >= 2) || Ygbm;



%% Simulation

% record jump times (inc. both types)
r    = nan(N,1);
r(1) = 0;

% indicator showing which process a jump time comes from
if saveZ
    Zfrom    = nan(N,1);
end

% record jump sizes (intensity jump)
if saveY
    Y_size = nan(N,1);
    
    if Ygbm
        Y_size(1) = parY(1);
    end
end

% record marked
if saveX
    X_size = nan(N,1);
end

% initialise lambda
lambda  = Y0;

% inter arrival time (cache)
s  = nan(1,2);


%% Simulation Loop for One Path
for j = 2:N
    
    % Sample jump times from background intensity 
    % (Will give immigrant)
    s(:,1) = - inv_mu .* log(rand);
    
    
    % Sample jump times for mutually-exciting process 
    % (Will give offspring)
    Smi = 1 + delta ./ lambda .* log(rand);
    Smi(Smi < 0) = 0;  % remove invalid Smi
    
    Smi = - inv_delta .* log(Smi);
    
    % Combine jump times
    s(:,2) = Smi;
    
    
    % Find minimum
    [s_min, I1] = min(s);
    %disp(s_min)
    
    % record event time
    r(j) = r(j-1) + s_min;
    
    i_star = I1 - 1;
    
    if saveZ
        Zfrom(j) = i_star;
    end
    
    % sample Y and update intensity
    YfromMstar = nan(1);
    if Yconst
        YfromMstar = parY;
    elseif Ygam
        YfromMstar = gamrnd(parY(1), parY(2));
    elseif Yexp
        YfromMstar = exprnd(parY);
    elseif Ygbm
        gbmMean = log(Y_size(j-1)) + parY(2) * s_min;
        gbmStdv = parY(3) * sqrt(s_min);
        YfromMstar = exp(normrnd(gbmMean, gbmStdv)); % lognormal
    elseif Ycln
        gbmMean = parY(2) * s_min - 1;
        gbmStdv = parY(3) * sqrt(s_min);
        YfromMstar = exp(normrnd(gbmMean, gbmStdv)); % lognormal
    end
       
    if saveY
        Y_size(j) = YfromMstar;       % matlab auto transpose
    end
    
    lambda = lambda .* exp(- delta * s_min) + YfromMstar;
    
    
    % sample X
    if saveX
        if Xconst
            X_size(j) = parX(1,1);
        elseif Xgam
            X_size(j) = gamrnd(parX(1,1,1), parX(1,1,2));
        elseif Xexp
            X_size(j) = exprnd(parX(1,1));
        elseif Xnorm
            X_size(j) = normrnd(parX(1,1,1), parX(1,1,2));
        end
    end
    
    % terminate loop if reach maturity time
    if r(j) > T_stop
        r(j) = nan;  % set r_j = nan, thus removed later
        break
    end
    
    
    % give error if pre-allocated vector is too short
    if j == N
        msg = 'Pre-allocated vector is too short!';
        error(msg);
    end
    
end


% trim nan from the data
ind_nan = isnan(r);  % indices of nan

r(ind_nan) = [];
if saveZ
    Zfrom(ind_nan) = [];
end
if saveX
    X_size(ind_nan) = [];
end
if saveY
    Y_size(ind_nan) = [];
end



%% Retrieve event times for the Hawkes process

% event times
t = cell(1);
if saveX
    X = cell(1);
end
if saveY
    Y = cell(1);
end
if saveZ
    Z = cell(1);
end


t{1} = r(2:end);
if saveX
    X{1} = X_size(2:end);
end
if saveY
    Y{1} = Y_size(2:end);
end
if saveZ
    Z{1} = Zfrom(2:end);
end


