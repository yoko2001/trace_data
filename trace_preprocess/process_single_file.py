import re
import os
import json

program_names = ['pagewalker']
target_program = "pagewalker"
trace_func_tpl = "folio_ws_chg"

def get_prog_pid(linestr):
    for prog in program_names:
        prog_pid_pattern = prog + '-' + '\d+'
        prog_pid = re.search(prog_pid_pattern, linestr, 0)
        if prog_pid:
            return prog_pid.group()
    return None
        
def parse_time(timestr): #get parsed time
    timestr = re.search("\d+\.\d+:", timestr, 0)
    if timestr:
        timestr = timestr.group()
        timestr = timestr.split(":")[0]
    time_f = float(timestr)
    return time_f

def parse_swap_and_direction(linestr):
    linestrret = linestr
    linestr = re.search("\[\S*\]left\[\S*\]", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr)[-1]
    else:
        exit(0)
    dir_and_left = linestr.split('left')
    dir = dir_and_left[0]
    left = dir_and_left[1]
    dir = dir[1:-1]
    left = left[1:-1]
    prio = dir[-1]
    if dir.startswith("RE<="):
        dir = "r"
    elif dir.startswith("EV=>"):
        dir = "e"
    else:
        print("err", dir)
        exit(0)
    leftover = int(left)
    return dir, prio, leftover, linestrret
def parse_entry(linestr):
    linestrret = linestr
    linestr = re.search("entry\[\S*\]", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr)[-1]
    else:
        exit(0)
        
    linestr = linestr.split("entry")[-1]
    linestr = linestr[1:-1]
    return linestr, linestrret

def parse_va(linestr):
    linestrret = linestr
    linestr = re.search("va\[\S*\]->", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr)[-1]
    else:
        exit(0)
        
    linestr = linestr.split("va")[-1]
    linestr = linestr[1:-3]
    return linestr, linestrret        

def parse_folio(linestr):
    linestrret = linestr
    linestr = re.search("folio@\[\S*\]\{", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr[0:-1])[-1]
    else:
        exit(0)
        
    linestr = linestr.split("folio@")[-1]
    linestr = linestr[1:-3]
    return linestr, linestrret 

def parse_folio(linestr):
    linestrret = linestr
    linestr = re.search("folio@\[\S*\]\{", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr[0:-1])[-1]
    else:
        exit(0)
        
    linestr = linestr.split("folio@")[-1]
    linestr = linestr[1:-3]
    return linestr, linestrret 

def parse_folio_bits(linestr):
    linestrret = linestr
    linestr = re.search("\{\[\S*\]ra\[\S*\]gen\[\S*\]\}", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr)[-1]
    else:
        exit(0)
    linestr = linestr[1:-1]
    linestr_sp =  linestr.split('ra')
    otherstr = linestr_sp[0]
    ra_gen = linestr_sp[-1].split('gen')
    ra = ra_gen[0][1:-1]
    gen = ra_gen[1][1:-1]
    swapprio = otherstr[1:-1]
    
    return swapprio, ra, gen, linestrret

def parse_memcg(linestr):
    linestrret = linestr
    linestr = re.search("\{memcg:\S*\}", linestr, 0)
    if linestr:
        linestr = linestr.group()
        linestrret = linestrret.split(linestr)[-1]
    else:
        exit(0)
        
    memcg = linestr[1:-1].split("memcg:")[-1]
    return memcg, linestrret 

def parse_single(totalstr, name):
    linestr = re.search(name + "\[\S*\]", totalstr, 0)
    if linestr:
        linestr = linestr.group()
    else:
        print("fail parsing single ", totalstr, name)
        exit(0)
    linestr = linestr.split(name)[-1]
    return int(linestr[1:-1])

def parse_line(linestr):
    if linestr.endswith("\n"):
        linestr = linestr[:-1]
    ret_list = []
    prog_pid = get_prog_pid(linestr)
    if prog_pid:
        if not prog_pid.startswith(target_program):
            return -1
    ret_list.append(prog_pid)
    #now we got prog_pid
    linestr = linestr.split(prog_pid)[-1]
    trace_func = re.search(trace_func_tpl + '\S*\:' , linestr, 0)
    if (trace_func):
        trace_func = trace_func.group()
    else:
        return None
    linestr_sp = linestr.split(trace_func)
    trace_func = trace_func[:-1]
    ret_list.append(trace_func)

    timestr = linestr_sp[0]
    linestr = linestr_sp[1]
    time_f = parse_time(timestr)
    ret_list.append(time_f)

    dir_, prio, left, linestr = parse_swap_and_direction(linestr)
    ret_list.append(dir_)
    ret_list.append(prio)
    ret_list.append(left)

    
    entry , linestr = parse_entry(linestr)
    ret_list.append(entry)

    va , linestr = parse_va(linestr)
    folio , linestr = parse_folio(linestr)
    ret_list.append(va)
    ret_list.append(folio)

    #{[s]ra[0]gen[-1]} [swapprio][low][high][readahead][gen] of folio bits
    swapprio, readahead_b, gen, linestr = parse_folio_bits(linestr)
    ret_list.append(swapprio)
    ret_list.append(readahead_b)
    ret_list.append(gen)

    memcg , linestr = parse_memcg(linestr)
    ret_list.append(memcg)

    linestr_spl = linestr.split(";")
    minseq_str = linestr_spl[0]
    minseq = parse_single(minseq_str, "min_seq")
    ref_str = linestr_spl[1]
    ref = parse_single(ref_str, "ref")
    tier_str = linestr_spl[2]
    tier = parse_single(tier_str, "tier")
    ret_list.append(minseq)
    ret_list.append(ref)
    ret_list.append(tier)

    print(ret_list)
    
    return ret_list

def parse_file_by_line(filepath, savepath):
    fp = open(filepath,"r")
    sp = open(savepath, "a")
    while True:
        line = fp.readline()
        if not line:      #等价于if line == "":
            break
        res = parse_line(line)
        if res:
            sp.write(str(res) + '\n')
    sp.close()
    fp.close()