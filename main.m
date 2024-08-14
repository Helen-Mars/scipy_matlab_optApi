%{ Get the path to the ProgramData directory%}
programDataPath = getenv('PROGRAMDATA');
setting_file = fullfile(programDataPath, 'EastWave', 'project_setting.json');

% Read and decode the JSON file
setting_data_str = fileread(setting_file);
setting_data = jsondecode(setting_data_str);

% Initialize the structure with the required data
need_setting_data = struct();
need_setting_data.work_directory = setting_data.work_catalog;
need_setting_data.solver_eastwave = setting_data.solver_wastwave;
need_setting_data.work_file = setting_data.work_file;
need_setting_data.np = setting_data.solver_param.thread_num;
need_setting_data.mxi_path = setting_data.mxi_path;
need_setting_data.opt_tool_path = setting_data.opt_tool_path;
need_setting_data.solver_param = setting_data.solver_param;

var_table = setting_data.var_table;
numElements = numel(var_table); 
var = cell(1, numElements);
var_value = struct();
x0 = zeros(1, numElements);

for i = 1:numElements
    var{i} = var_table{i}{1};  % 提取每个子对象的第一个元素
    var_value.(var_table{i}{1}) = str2num(var_table{i}{3});
    x0(i) = str2double(var_table{i}{3});
end

need_setting_data.var = var;

% Define the objective function, using the initialized data
objective = @(x) objectivefunc1(x, need_setting_data);

% x0 = [10, 100];
disp('x0= ');
disp(x0);

% global func_evaluate_count;
% func_evaluate_count = 0;
if setting_data.matlab_opt_id == 0
    options = optimset('Display','iter',...
                       'PlotFcns', @optimplotfval,...
                       'MaxFunEvals', int32(setting_data.opt_setting.MaxFunEvals), ...
                       'MaxIter', int32(setting_data.opt_setting.MaxIter), ...
                       'TolFun', setting_data.opt_setting.TolFun, ...
                       'TolX', setting_data.opt_setting.TolX);    

    [x, fval, exitflag, output] = fminsearch(objective, x0, options);

    fprintf('x: %.4f\n', x);
    fprintf('fval: %.4f\n', fval);
    fprintf('exitflag: %d\n', exitflag);
    fprintf('output: %s\n', output.message);

elseif setting_data.matlab_opt_id == 1
    disp("pass")
elseif setting_data.matlab_opt_id == 2
    disp("pass")
end

system('pause');


