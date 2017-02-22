import io
import itertools

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

#adlist={parent:[list of kids]}

layertype={}
layerprop={}
modelinputs={}
modeloutputs={}
connectionslist=[]
adlist={}
primitivemappings={}
implementednodes=set()
implementedmodels=set()

def merge_notebooks(filenames):
    merged = None
    for fname in filenames:
        with io.open('temp/' + fname + '.ipynb', 'r', encoding='utf-8') as f:
            nb = nbf.read(f, as_version=4)
        if merged is None:
            merged = nb
        else:
            merged.cells.extend(nb.cells)
    if not hasattr(merged.metadata, 'name'):
        merged.metadata.name = ''
    merged.metadata.name += "_merged"
    with open('notebooks/notebook.ipynb', 'w') as f:
        nbf.write(merged, f)


def updateadj(adj,key,value):

    if key not in adj.keys():
        adj[key] = set()

    adj[key].add(value)


def readconfig(configfile, mappings):
    f = open(configfile)
    line = f.readline()
    position = ''
    layerdone = 0

    for line in f:
        if layerdone == 0:
            line = line.strip()
            if line == 'LAYERS:':
                position = line
                layerdone = 1
                continue
            else:
                continue

        if position == 'LAYERS:':
            line = line.strip().replace(' ', '')
            if line == 'CONNECTIONS:':
                position = line
                continue
            if line == '':
                continue
            line = line.split('\t')
            name = line[0]
            type = line[1]
            if len(line) > 2:
                properties = line[2]
            layertype[name] = type
            layerprop[name] = properties.replace('{', '').replace('}', '')

        if position == 'CONNECTIONS:':
            line = line.strip().replace(' ', '')
            if line == 'MODELS:':
                position = line
                continue
            if line == '':
                continue
            line = line.strip().replace(' ', '')
            connectionslist.append(line)

        if position == 'MODELS:':
            line = line.strip().replace(' ', '')
            if line == '':
                continue
            line = line.split(':')
            name = line[1]
            line = line[0].split('->')
            input = line[0]
            input = input.split(',')
            output = line[1]
            modelinputs[name] = input
            modeloutputs[name] = output

    f.close()

    mappings=open(mappings,'r')
    for line in mappings:
        line = line.strip().replace(' ', '')
        line=line.split(':')
        primitivemappings[line[0]]=line[1]


    for modelprimes in modelinputs.keys():
        primitivemappings[modelprimes]=modelprimes



def createadjecency():


    for line in connectionslist:
        line=line.strip()
        if ':' in line:
            inputs = line.split(':')[0].split('+')
            output=line.split(':')[1]
            for input in inputs:
                updateadj(adlist,output,input)


        else:
            inputs = line.split('>')
            for i in range(0,len(inputs)-1):
                updateadj(adlist,inputs[i+1],inputs[i])

    tops=list(set(adlist.keys()) - set(list(itertools.chain.from_iterable(adlist.values()))))
    return tops








def producecode(myname):

    code = ''

    if myname in implementednodes:
        return code
    else:
        implementednodes.add(myname)



    type=layertype[myname]
    properties=layerprop[myname]
    modelname=''

    if '@@' in type:
        type = type.replace('@@', '')
        modelname=type

        codetemp = producecode(modeloutputs[modelname])
        code += '\n' + codetemp

        code +='\n'+type+' = Model('
        code +='output = '+modeloutputs[modelname]+' ,input = ['
        implementedmodels.add(type)

        for input in modelinputs[modelname]:
            code +=input+','

        code=code[:-1]
        code += '], name = \'' + type + '\')'



    elif '@' in type:
        type=type.replace('@','')
        properties = layerprop[type]
        type=layertype[type]

    type=primitivemappings[type]



    if myname in adlist.keys():
        kids=adlist[myname]

    else:
        if modelname!='':
            code += '\n'+myname + ' = ' + type
        else:
            code+=myname+' = '+type+'('+properties+')'
        return code

    for kid in kids:
        codetemp=producecode(kid)
        code+='\n'+codetemp


    if len(kids)==1:
        if modelname!='':
            code += '\n' + myname + ' = ' + type +'(' + list(kids)[0] + ')'
        else:
            code += '\n' + myname + ' = ' + type + '(' + properties + ')' + '(' + list(kids)[0] + ')'
    else:
        if type=='merge':
            code += '\n' + myname + ' = ' + type + '(['
            for kid in kids:
                code+=kid+','
            code=code[:-1]
            code+=']'
            code+=','+properties
            code+=')'
        else:
            code += '\n' + myname + ' = ' + type
            kids=list(kids)
            if kids[0] in modeloutputs.keys():
                code += '('+ kids[0] +')'+'('+ kids[1] +')'
            else:
                code += '(' + kids[1] + ')' + '(' + kids[0] + ')'


    return code



def start():
    autocode = open('gencode/auto.py', 'w')

    imports='from keras.layers import Convolution2D, MaxPooling2D, Flatten' \
            '\nfrom keras.layers import Input, LSTM, Embedding, Dense, merge' \
            '\nfrom keras.models import Model, Sequential' \
            '\nfrom keras.utils.visualize_util import plot' \
            '\nfrom keras.layers import TimeDistributed'

    readconfig('configuration/config', 'configuration/mappings')
    tops=createadjecency()


    code=''
    for modeloutput in modeloutputs.values():
        code+=producecode(modeloutput)

    remainingmodels=set(modeloutputs.keys())-implementedmodels


    for model in remainingmodels:
        code += '\n' + model + '=Model('
        code += 'output = ' + modeloutputs[model] + ' ,input = ['
        for input in modelinputs[model]:
            code += input + ','

        code = code[:-1]
        code += '])'

    plotdir = 'plots/'
    for model in modeloutputs.keys():
        code += '\n' + 'plot(' + model + ', to_file = \'' + plotdir + model + '.png\', show_layer_names=True, show_shapes=True)'

    # print code

    autocode.write(imports)
    autocode.write(code)
    autocode.close()



    code=imports+code
    plotdir = 'plots/'
    imagetemp1="from IPython.display import Image\n"
    imagetemp2 = "Image(filename=\'" + plotdir
    imagetemp3="\', embed=True,format='png')"

    nb = nbf.v4.new_notebook()

    text = "This is an auto-generated keras model code."

    nb['cells'] = [nbf.v4.new_markdown_cell(text),
                   nbf.v4.new_code_cell(code),
                   ]

    ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
    ep.preprocess(nb,{})

    with open('temp/code.ipynb', 'w') as f:
        nbf.write(nb, f)


    for model in modeloutputs.keys():
        outputcode = imagetemp1+imagetemp2 + model + '.png' + imagetemp3 + '\n'
        nbout = nbf.v4.new_notebook()
        nbout['cells'] = [(nbf.v4.new_code_cell(outputcode))]
        ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
        ep.preprocess(nbout, {})
        with open('temp/' + model + '.ipynb', 'w') as f:
            nbf.write(nbout, f)

    merge=[]
    merge.append('code')
    merge.extend(modeloutputs.keys())

    merge_notebooks(merge)

if __name__ == '__main__':

    start()






