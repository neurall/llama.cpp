import re,sys,importlib.util,shutil
sys.argv=['run.py']
sp=importlib.util.spec_from_file_location('run','/p/bw/wt-main/fork/run.py'); run=importlib.util.module_from_spec(sp); sp.loader.exec_module(run)
ANSI=re.compile(r'\x1b\[([0-9;?]*)([A-Za-z])')
def emulate(text,W,H):
    """tiny terminal: auto-wrap at W, scroll at H, cursor up/erase/home/clear"""
    scr=[[]]; r=0; c=0; scroll=0
    rows=[""]  # list of strings; visible rows = last H of rows? model with explicit top
    lines=[""]; cur=0; col=0   # lines: all rows since start; cur: cursor row index (absolute); top = max(0,len(lines)-H) tracked as scrolled
    top=0
    def ensure(n):
        while len(lines)<=n: lines.append("")
    i=0
    while i<len(text):
        m=ANSI.match(text,i)
        if m:
            i=m.end(); a,k=m.group(1),m.group(2)
            if k=='A': cur=max(top,cur-int(a or 1))
            elif k=='J':
                if a in('','0'): ensure(cur); lines[cur]=lines[cur][:col]; del lines[cur+1:]
                elif a=='2': del lines[top:]; ensure(top); cur=top; col=0; lines[top:]=[""]
            elif k=='H': cur=top; col=0
            continue
        ch=text[i]; i+=1
        if ch=='\n':
            cur+=1; col=0; ensure(cur)
        elif ch=='\r': col=0
        else:
            if col>=W: cur+=1; col=0; ensure(cur)
            ensure(cur); lines[cur]=lines[cur][:col]+ch+lines[cur][col+1:]; col+=1
        if cur-top>=H: top=cur-H+1
    return lines,top
def test(logpath,W,H,filt,limit=400000):
    data=open(logpath,'rb').read()[-limit:]
    import re as _re
    if filt: data=_re.sub(rb'(\x1b\[\d+A\x1b\[J)',lambda m: m.group(1)+b'\x1e',data)   # markers as the new daemon writes them (test: every text after a clear counts as the frame)
    # start at a clean point
    if filt:
        vf=run.ViewFilter()
        import os
        shutil.get_terminal_size=lambda fb=None: os.terminal_size((W,H))
        run.shutil.get_terminal_size=shutil.get_terminal_size
        out=""
        for i in range(0,len(data),65536): out+=vf.feed(data[i:i+65536])
    else: out=data.decode('utf8','replace')
    lines,top=emulate(out,W,H)
    vis=lines[top:]
    cpu=sum(1 for l in vis if l.startswith('moe cache') or l.startswith('cpu '))
    return cpu,len(vis),top
log='/p/bw/data/q5h/run-list-1026-102349.log'
for W,H in ((100,45),(120,60),(160,70),(110,30)):
    for filt in (False,True):
        print(W,H,'filter' if filt else 'raw   ', 'cpu rows visible, visible rows, scrolled:',test(log,W,H,filt))
