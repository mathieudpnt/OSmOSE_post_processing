function [dtFormat] = getStrFormat(dateTimeStr)

% Define the possible formats
formats = {'yyMMddHHmmss', 'yyyy-MM-dd_HH-mm-ss'};

% Try to create a datetime object using each format
for i = 1:numel(formats)
    try
        % Use the datetime function to create a datetime object with the current format
        dt = datetime(dateTimeStr, 'InputFormat', formats{i});
        
        % If the datetime object was created successfully, break out of the loop and proceed to the next step
        dtFormat = formats{i};
        break;
    catch
        % If the datetime function could not create a datetime object with the current format, continue to the next format
        continue;
    end
end

if ~exist('dtFormat', 'var')
    disp('Datetime format not recognised')
end

end

