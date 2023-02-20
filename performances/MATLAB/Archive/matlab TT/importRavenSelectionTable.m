function SelectionTable = importRavenSelectionTable(filename)

dataLines = [1, Inf];

opts = delimitedTextImportOptions("NumVariables", 8);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["Selection", "View", "Channel", "BeginTimes", "EndTimes", "LowFreq", "HighFreq", "Type"];
opts.SelectedVariableNames = ["Selection", "View", "Channel", "BeginTimes", "EndTimes", "LowFreq", "HighFreq", "Type"];
opts.VariableTypes = ["double", "string", "double", "double", "double", "double", "double", "string"];

% Specify file level properties
opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";


% Specify variable properties
opts = setvaropts(opts, ["View", "Type"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Selection", "View", "Channel", "BeginTimes", "EndTimes", "LowFreq", "HighFreq", "Type"], "EmptyFieldRule", "auto");

% Import the data
SelectionTable = readtable(filename, opts);

end