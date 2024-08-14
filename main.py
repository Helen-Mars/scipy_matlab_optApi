# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import subprocess as sp
from numpy import (finfo, sqrt)
from scipy import optimize
from collections import OrderedDict
import logging
# import inspect


# 定义函数执行次数
func_evaluate_count= 0
program_data_path = os.environ.get('PROGRAMDATA')
setting_path = os.path.join(program_data_path, 'EastWave', 'project_setting.json')

with open(setting_path, encoding='utf-8') as json_setting:
    json_setting = json.load(json_setting)

#print(json_setting["opt_tool_path"])

current_path = json_setting["opt_tool_path"]
#current_path = "E:\\ew_project\\ew_bin\\ew7\\scripting\\data\\toolbox\\toolbox.opt_tool"
os.chdir(current_path)


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s -%(levelname)s - %(message)s')
file_handle = logging.FileHandler(json_setting['work_catalog'] +'log.txt')
file_handle.setLevel(logging.INFO)
file_handle.setFormatter(logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s'))

logger = logging.getLogger("opt_logger")
logger.addHandler(file_handle)

_epsilon = sqrt(finfo(float).eps)

var_table = json_setting['var_table']


# 重构一些接口参数
x0 = [0]*len(var_table)
bounds = [0]*len(var_table)
var = OrderedDict()
variable = [0]*len(var_table)

for index, item in enumerate(var_table):
    var[item[0]] = eval(item[2])
    x0[index] = eval(item[2])
    bounds[index] = tuple(eval(item[1]))
    variable[index] = item[0]


work_catalog = json_setting['work_catalog']
with open(work_catalog+"project_opt_param.json", 'w') as f:
    json.dump(var, f, indent=4)



#构造边界
# =============================================================================
# try:  
#     bounds2 = json_setting['bounds']
#     bounds = [tuple(i) for i in bounds2]
# except Exception:
#     logger.warning("Build bounds failure")
# =============================================================================

#构造约束条件函数
def create_cons(equal_cons, inequal_cons):
    cons_out = []  
    if equal_cons:            
        len_equal_cons = len(equal_cons)
        func_equal_names = ["h"+str(i) for i in range(len_equal_cons)]
        func_equal = {}
        count1 = 0
        for func_equal_name in func_equal_names:
            obj_str = equal_cons[count1]
            func_equal[func_equal_name] = lambda x: eval(obj_str)
            count1 += 1
        for key, value in func_equal.items():
            temp_dict = {}
            temp_dict['type'] = 'eq'
            temp_dict['fun'] = value
            cons_out.append(temp_dict)      
            
    if inequal_cons:    
        len_inequal_cons = len(inequal_cons)
        func_inequal_names = ["g"+str(i) for i in range(len_inequal_cons)]

        func_inequal = {}
        count2 = 0
        for func_inequal_name in func_inequal_names:
            obj_str1 = inequal_cons[count2]
            func_inequal[func_inequal_name] = lambda x: eval(obj_str1)
            count2 += 1
        for key, value in func_inequal.items():
            temp_dict = {}
            temp_dict['type'] = 'ineq'
            temp_dict['fun'] = value
            cons_out.append(temp_dict) 
                 
    return tuple(cons_out)

#定义写入文件的函数
def write_txt(list_value):
    with open(json_setting['work_catalog'] + 'opt_result.txt', 'a', encoding="utf-8") as f:
        f.write('\n')    
        for item in list_value:
            f.write("{:<20}".format(item))
    

solver_param = json_setting['solver_param']
#定义优化目标函数
def aim_func(x):
    
    global func_evaluate_count
    
    with open(work_catalog +'project_opt_param.json', "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file, object_pairs_hook=OrderedDict) 
        
    reordered_json_data = OrderedDict((key, json_data[key]) for key in variable)
       
    count = 0
    
    for k,v in reordered_json_data.items():
        reordered_json_data[k] = x[count]
        count += 1
        
    with open(json_setting['work_catalog']+'project_opt_param.json', "w", encoding='utf-8') as to_json_file:
        json.dump(reordered_json_data, to_json_file, indent=4)     
        
        
    run_param = [json_setting["solver_wastwave"],\
                          "-set-var-by-json",json_setting['work_catalog']+"project_opt_param.json",\
                          json_setting["work_file"],\
                          "-np",solver_param['thread_num'],\
                          "-tee", json_setting['work_catalog']+'solver_log.txt']
    run_param.append(solver_param['vectorization_opt']) if solver_param['vectorization_opt'] != "" else None
    run_param.append(solver_param['fp_precision']) if solver_param['fp_precision'] != "" else None  
    run_param.append(solver_param['arguments']) if solver_param['arguments'] != "" else None    

     
    return_flag = sp.run(run_param, shell=True, encoding="utf-8")
        
    if return_flag.returncode != 0:
        raise Exception("Eastwave solver fialed to process ewp2_file!")

    return_mxd_flag = sp.run([json_setting["mxi_path"], "deal_mxd.mx"], shell=True, encoding="utf-8")
    if return_mxd_flag.returncode != 0:
        raise Exception("mxi failed to process mxd_file!")
    
    func_evaluate_count += 1
    logger.info("Finished {} times objective computation in total.".format(func_evaluate_count))
    
    aim_value = np.loadtxt(json_setting['work_catalog']+'output_data.txt')
    # print(aim_value)
    
    temp_opt_value = (np.hstack((x, aim_value))).tolist()

    write_txt(temp_opt_value)
    
    return aim_value


# 定义优化函数
def optimize_func(solver_id, bounds, cons, enable_cons=0): 
    opt_result = dict()
    opt_result['result'] = None
    options = {'disp': True}
    
    if solver_id == 0: 
        option = json_setting['opt_setting']['option']

        n1 = eval(json_setting['opt_setting']['sampling_num'])
        iters1 = eval(json_setting['opt_setting']['iter_num'])
        workers = eval(json_setting['opt_setting']['workers'])
        sampling_method = json_setting['opt_setting']['sampling_method']
        
        option1 = {k:eval(v) for k,v in option.items()}
        options.update(option1)
        
        f_tol_local = eval(json_setting['opt_setting']['f_tol_local'])
        minimizer_kwargs = {'options': {'ftol': f_tol_local}}

        constraints1 = None if enable_cons==0 else cons
        opt_result['result'] = optimize.shgo(aim_func, bounds, constraints=constraints1, n=n1,\
                                             iters=iters1, minimizer_kwargs= minimizer_kwargs, \
                                             options=options, sampling_method=sampling_method, workers= workers)
       
    elif solver_id == 1:
         opt_param = json_setting['opt_setting']
         opt_param = {k:eval(v) for k,v in opt_param.items()}
         # x0 = np.array(x0)
        
         opt_result['result'] = optimize.dual_annealing(aim_func, bounds, **opt_param)
        
    elif solver_id == 2:
        opt_param = json_setting['opt_setting']
        # opt_param = {k:eval(v) if not (isinstance(v, str) and v.startswith("(")) else v for k,v in opt_param.items()}
        for k,v in opt_param.items():
            try:
                evaluated_value = eval(v)
                opt_param[k] = evaluated_value
            except (NameError, SyntaxError, TypeError):
                opt_param[k] = v
        
        opt_param.update(options)
        opt_result['result'] = optimize.differential_evolution(aim_func, bounds, x0=x0, **opt_param)
        
    elif solver_id == 3:  
        opt_result['result'] = optimize.basinhopping(aim_func, bounds)
        
    elif solver_id == 4:
        opt_setting = json_setting['opt_setting']
        option1 = {k:eval(v) for k,v in opt_setting.items()}
        options.update(option1)
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='Nelder-Mead', bounds=bounds,\
                                                 options=options)
    elif solver_id == 5:
        opt_setting = json_setting['opt_setting']
        option1 = {k:eval(v) for k,v in opt_setting.items()}
        options.update(option1)       
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='Powell', bounds=bounds,\
                                                 options=options)
    elif solver_id == 6:
        gtol1 = eval(json_setting['opt_setting']['gtol'])
        maxiter1 = eval(json_setting['opt_setting']['maxiter'])
        eps1 = eval(json_setting['opt_setting']['eps'])
        options.update({'gtol':gtol1, 'eps':eps1, 'maxiter':maxiter1})
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='CG', options=options)

    elif solver_id == 7:
        gtol1 = eval(json_setting['opt_setting']['gtol'])
        maxiter1 = eval(json_setting['opt_setting']['maxiter'])
        eps1 = eval(json_setting['opt_setting']['eps'])
        xrtol1 = eval(json_setting['opt_setting']['xrtol'])
        options.update({'gtol':gtol1, 'eps':eps1, 'maxiter':maxiter1, 'return_all':True, 'xrtol':xrtol1})  
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='BFGS', options=options)
        
    elif solver_id == 8:
        gtol1 = eval(json_setting['opt_setting']['gtol'])
        ftol1 = eval(json_setting['opt_setting']['ftol'])
        maxcor1 = eval(json_setting['opt_setting']['maxcor'])
        maxiter1 = eval(json_setting['opt_setting']['maxiter'])
        maxfun1 = eval(json_setting['opt_setting']['maxfun'])
        maxls1 = eval(json_setting['opt_setting']['maxls'])
        eps1 = eval(json_setting['opt_setting']['eps'])
        
        options.update({'gtol':gtol1, 'ftol': ftol1, 'eps':eps1, 'maxiter':maxiter1, 'maxcor':maxcor1,\
                        'maxfun': maxfun1, 'maxls':maxls1})  
        
        opt_result['result'] = optimize.minimize(aim_func, x0, bounds=bounds, method='L-BFGS-B', options=options)
        
    elif solver_id == 9:
        eps = eval(json_setting['opt_setting']['eps'])
        scale = eval(json_setting['opt_setting']['scale'])
        offset = eval(json_setting['opt_setting']['offset'])
        maxCGit = eval(json_setting['opt_setting']['maxCGit'])
        eta = eval(json_setting['opt_setting']['eta'])
        stepmx = eval(json_setting['opt_setting']['stepmx'])
        accuracy = eval(json_setting['opt_setting']['accuracy'])
        minfev = eval(json_setting['opt_setting']['minfev'])
        ftol = eval(json_setting['opt_setting']['ftol'])
        gtol = eval(json_setting['opt_setting']['gtol'])
        rescale = eval(json_setting['opt_setting']['rescale'])
        maxfun = eval(json_setting['opt_setting']['maxfun'])
        
        option1 = {'eps': eps, 'scale':scale, 'offset':offset, 'maxCGit': maxCGit, 'eta': eta,\
                  'stepmx': stepmx, 'accuracy': accuracy, 'minfev': minfev, 'ftol': ftol, 'gtol': gtol,\
                      'rescale': rescale, 'maxfun': maxfun}
        options.update(option1)
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='TNC', bounds=bounds, options=options)
        
    elif solver_id == 10: 
        catol = eval(json_setting['opt_setting']['catol'])
        maxiter = eval(json_setting['opt_setting']['maxiter'])   
        rhobeg = eval(json_setting['opt_setting']['rhobeg'])
        tol = eval(json_setting['opt_setting']['tol'])
        
        constraints = None if enable_cons==0 else cons        
        
        option1 = {'catol': catol, 'maxiter': maxiter, 'rhobeg': rhobeg, 'tol':tol}
        options.update(option1)
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='COBYLA', bounds=bounds, options=options, constraints=constraints)

    elif solver_id == 11: 
        ftol = eval(json_setting['opt_setting']['ftol'])
        maxiter = eval(json_setting['opt_setting']['maxiter'])   
        eps = eval(json_setting['opt_setting']['eps']) 
        
        eps = _epsilon if eps==None else eps
        option1 = {'ftol':ftol, 'maxiter': maxiter, 'eps': eps}
        options.update(option1)
        
        constraints = None if enable_cons==0 else cons    
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method= 'SLSQP', bounds=bounds, \
                                                constraints= constraints, options=options)        
    elif solver_id == 12:
        opt_setting = json_setting['opt_setting']
        option1 = {k:eval(v) for k,v in opt_setting.items()}
        options.update(option1)
        
        constraints = None if enable_cons==0 else cons 
        
        opt_result['result'] = optimize.minimize(aim_func, x0, method='trust-constr', bounds=bounds, \
                                                constraints= constraints, options=options) 
            
    elif solver_id == 13:
        opt_setting = json_setting['opt_setting']
        option1 = {k:eval(v) for k,v in opt_setting.items()}

        opt_result['result'] = optimize.direct(aim_func, bounds=bounds, **option1) 
        
    return opt_result



def main():
    
    first_row = list(variable)
    first_row.append("func")
    
    file = open(json_setting['work_catalog'] + 'opt_result.txt', 'w')
    file.truncate(0)
    file.close()
    
    equal_cons = json_setting['equal_cons']
    inequal_cons = json_setting['inequal_cons']
    
    #构造约束条件
    try:
        cons = create_cons(equal_cons, inequal_cons)
    except Exception:
        logger.warning("Build Constraints failure")
        raise
    
    write_txt(first_row) 
    
    
    enable_constraint = json_setting['enable_cons']
    solver_id = json_setting['solver_type.id']
    opt_result = optimize_func(solver_id, bounds, cons, enable_cons=enable_constraint)
    logger.info(opt_result['result'])
    
    with open(json_setting['work_catalog'] + 'opt_result.txt', 'a', encoding="utf-8") as file:
        file.write('\n\n' + str(opt_result['result']))
    
if __name__ == '__main__':
    main()
    os.system("pause")





