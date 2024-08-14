function f = objectivefunc1(x, need_setting_data)

var = need_setting_data.var;
var_value = x;

varStruc = struct();
for i = 1:length(var)
    varStruc.(var{i}) = var_value(i);
end

jsonString = jsonencode(varStruc);
opt_json_file = fullfile(need_setting_data.work_directory, "project_opt_param.json");
fileID = fopen(opt_json_file, 'w');
fprintf(fileID, '%s', jsonString);
fclose(fileID);

paramJsonfile = fullfile(need_setting_data.work_directory, "project_opt_param.json");

runParam = {need_setting_data.work_file, ...
    '-set-var-by-json', paramJsonfile, ...
    '-np', need_setting_data.np, ...
    '-tee', fullfile(need_setting_data.work_directory, 'solver_log.txt')};

if ~isempty(need_setting_data.solver_param.vectorization_opt)
    runParam{end+1} = need_setting_data.solver_param.vectorization_opt;
elseif ~isempty(need_setting_data.solver_param.fp_precision)
    runParam{end+1} = need_setting_data.solver_param.fp_precision;
elseif ~isempty(need_setting_data.solver_param.arguments)
    runParam{end+1} = need_setting_data.solver_param.arguments;
end

runParamStr = cellfun(@(x) char(x), runParam, 'UniformOutput', false);

command_solver = sprintf('"%s" %s', need_setting_data.solver_eastwave, strjoin(runParamStr, ' '));

[status, ~] = system(command_solver);

if status ~= 0
    error('Command failed with status %d: %s', status);
end

command_mxi = sprintf('"%s" "%s"', need_setting_data.mxi_path, fullfile(need_setting_data.opt_tool_path, "deal_mxd.mx"));

[status1, ~] = system(command_mxi);

if status1 ~= 0
    error('Command failed with status %d: %s', status);
end

% global func_evaluate_count;
% func_evaluate_count = func_evaluate_count+1;

% fprintf('Finished %d times objective computation in total.\n', func_evaluate_count);

opt_value = load(fullfile(need_setting_data.work_directory, "output_data.txt"));
if ~isfloat(opt_value)
    error("Didn't get a effective optimization value.");
end    

f = opt_value;