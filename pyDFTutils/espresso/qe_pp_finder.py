#!/usr/bin/env python
"""
This python module is used to:
1. read pseudopotential files, gather the information and put them into a sqlite3 database file.
2. find pseudopotentials from the database.
"""
import lxml.etree as ET
import os
import sys
import sqlite3
import argparse
import glob
import tempfile
import re

def phase_qe_name(filename):
    xc_name=filename.split('.')[1].split(('-'))[0]
    if xc_name.startswith(('rel','star')):
        xc_name=filename.split('.')[1].split(('-'))[1]
    #print element,xc_name,filename
    return xc_name

def phase_qe_pp(filename):
    xc_name=phase_qe_name(os.path.basename(filename))
    text=open(filename,'r').readlines()[:300]
    suggested_ecutwfc=None
    suggested_ecutrho=None
    full_rel=1
    config=None
    #if text[0].startswith(r'<PP_INFO>'):
    #    return
    head_text=''
    inblock=False
    for line in text:
        ##Try to read the suggested cutoffs. If failed, set to -100.
        if line.find('Suggested minimum cutoff for wavefunctions')!=-1:
            suggested_ecutwfc=float(line.strip().split()[-2])
        if line.find('Suggested minimum cutoff for charge density')!=-1:
            suggested_ecutrho=float(line.strip().replace(':','  ').split()[-2])
        if line.find('rel=')!=-1:
            full_rel=int(line.strip().strip(',').split('=')[-1])
        if line.find('config=')!=-1:
            config=line.strip().strip(',').split('=')[-1]

        if line.strip().startswith(r'<PP_HEADER'):
            inblock=True
        if inblock and line.find('=') !=-1:
            head_text+=line
        if line.strip().endswith(r'/>'):
            inblock=False
    parser = ET.XMLParser(encoding='utf-8',recover=True)
    root=ET.fromstring(head_text,parser=parser)

    symbol=root.attrib['element'].strip()
    valence=int(float(root.attrib['z_valence']))
    #xc_name=root.attrib['functional']/opt/src/pp/pslib/pslibrary.1.0.0/rel-pbe/PSEUDOPOTENTIALS/Sb.rel-pb
    xc_type='NC'
    if root.attrib['is_ultrasoft']=='T':
        xc_type='US'
    if root.attrib['is_paw']=='T':
        xc_type='PAW'
    if root.attrib['is_coulomb']=='T':
        xc_type='COULOMB'
    return symbol,valence,xc_type,xc_name,filename,suggested_ecutwfc,suggested_ecutrho,full_rel,config


def phase_all():
    dbfile='./qeppdb.db'
    #dbfile='./ppdb.db'
    if os.path.exists(dbfile):
        os.remove(dbfile)
    mydb=sqlite3.connect(dbfile)
    mycur=mydb.cursor()
    mycur.execute("""
    create table pptable (
    symbol text,
    valence integer,
    pp_type text,
    pp_name text,
    path text,
    ecutwfc real,
    ecutrho real,
    full_rel integer,
    config text
    )
    """)
    for path,v,filenames in os.walk('/opt/src/pp/pslib/pslibrary.1.0.0',followlinks=True):
        for filename in filenames:
	    print path
            if filename.endswith('.UPF'):
                fullname=os.path.join(path,filename)
                print fullname
                vals= phase_qe_pp(fullname)
                print vals
                mycur.execute("""
                insert into pptable values(
                ?,?,?,?,?,?,?,?,?)
                """, vals)
    mydb.commit()
    lines=mycur.execute("select * from pptable")
    for line in lines:
        print map(str,line)
    mydb.close()


def find_pp(symbol,minval=0,pp_type=None,pp_name=None,rel=None):
    info=[]
    #mydb=sqlite3.connect('./ppdb.db')
    mydb=sqlite3.connect('/opt/src/pp/qeppdb.db')
    mycur=mydb.cursor()
    command="select * from pptable where symbol='%s' and valence > %d"%(str(symbol) ,minval)
    #command="select symbol,pp_type,pp_name,valence,ecutwfc,ecutrho from pptable where symbol='%s' and valence > %d"%(bytes(symbol),minval)
    if pp_type is not None:
        command+= " and pp_type = '%s'"%pp_type
    if pp_name is not None:
        command+= " and pp_name = '%s'"%pp_name
    if rel is not None:
        command+= " and full_rel = '%s'"%rel
    print command
    lines=mycur.execute(command)
    byte_to_str=lambda x: bytes(x) if type(x) not in( int, float) else x
    for line in lines:
        info.append( map(byte_to_str,line))
    mydb.close()
    return info

def find_pp_s(symbol,pp_type='US',xc_name='pz',rel=1,label='soft'):
    """
    find the espresso pseudopotential in the pslib.
    symbol: element name
    pp_type: 'US' | 'PAW' |'NC'. currently no NC pp .
    xc_name:'pz' | 'pbe'
    rel: 1 | 2 , 1: scalar realistive. 2. full
    label: 'soft' |'hard'. If two pp are present. choose the softer/harder one
    returns the [symbol, valence, pp_type, xc_name, path, ecutwav, ecutrho, config]
    """
    pp_infos=find_pp(symbol,pp_type=pp_type,pp_name=xc_name,rel=rel)
    if len(pp_infos)==1:
        return pp_infos[0]
    elif len(pp_infos)==2:
        a,b=pp_infos
        if a[-4]<b[-4]:
            soft,hard=a,b
        else:
            soft,hard=b,a
        if label=='soft':
            return soft
        elif label=='hard':
            return hard
        else:
            raise ValueError('label should be soft or hard')


def find():
    parser=argparse.ArgumentParser()
    #parser.add_argument('-g','--gen', help="Generate the pp database")
    parser.add_argument("symbol", help="The symbol of the element")
    parser.add_argument('-t','--type', help="The type of the pp,LDA of GGA")
    parser.add_argument('-x','--xc',help="The name of the pp, eg. PBE, PW, rPBE")
    parser.add_argument('-m','--minval',help="the min valence",default=0)

    args=parser.parse_args()
    
    return find_pp(args.symbol,pp_type=args.type,pp_name=args.xc,minval=args.minval)


if __name__=='__main__':
    #phase_all()
    #phase_vasp_pp('/opt/src/pp/vasp/paw/Mo_pv/POTCAR.Z')
    for x in find():
        print(x)
    #print find_pp_s('O','PAW','pz')

    """
    for f in os.listdir('../'):
        if f.endswith('.UPF'):
            phase_qe_name(f )
            phase_qe_old_pp('../%s'%f)
            print f
            phase_qe_pp('../%s'%f)
    """
