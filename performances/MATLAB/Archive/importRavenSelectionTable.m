function SelectionTable = importRavenSelectionTableNEW(filename)

dataLines = [1, Inf];

opts = delimitedTextImportOptions("NumVariables", 7);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["Var1", "Var2", "Var3", "BeginTimes", "EndTimes", "Var6", "Var7"];
opts.SelectedVariableNames = ["BeginTimes", "EndTimes"];
opts.VariableTypes = ["string", "string", "string", "double", "double", "string", "string"];

% Specify file level properties
opts.ImportErrorRule = "omitrow";
opts.MissingRule = "omitrow";
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";


% Specify variable properties
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var6", "Var7"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["Var1", "Var2", "Var3", "Var6", "Var7"], "EmptyFieldRule", "auto");

% Import the data
SelectionTable = readtable(filename, opts);

%% Convert to output type
SelectionTable = table2array(SelectionTable);
end