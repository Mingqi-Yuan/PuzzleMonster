class Node:
    def __init__(self,parent,state,degree=0,h=0,f=0):
        self.parent=parent
        self.state=state
        self.degree=degree
        self.h = h 
        self.f = f


class ASTARAgent:
    def __init__(self,originaNode,targetNode,MaxDegree,length):
        self.originNode=originaNode
        self.targetNode=targetNode
        self.open=[self.originNode]
        self.close=[self.originNode]
        self.spce=[-3,3,-1,1] #上下左右四个移动方向
        self.MaxDegree=MaxDegree  #深度限制，到达此深度未找到解便返回
        self.length=length

    #判断是否有解
    def hasSolve(self):
        targetVer=self.getreVersNum(self.targetNode.state)
        orinateVer=self.getreVersNum(self.originNode.state)
        if(targetVer%2!=orinateVer%2):
            return False
        else:
            return True
    #获取逆序数
    def getreVersNum(self,state):
        sum=0
        for i in range(0,len(state)):
            if(state[i]==0):
                continue
            else:
                for j in range(0,i):
                    if(state[j]>state[i]):
                        sum+=1
        return sum

    def copyArray(self,state):
        arr=[]
        return arr+state

    def isInTable(self,node,table):
        for i in table:
            if i.state==node.state and i.degree==node.degree:
                return True
        return False

    def showLine(self):
        endState=self.open[-1]
        road=[endState]
        while(True):
            if(endState.parent):
                endState=endState.parent
                road.append(endState)
            else:
                break
        road.reverse()
        for j in road:
            for i in range(0,3):
                print(j.state[i*3:i*3+3])

            print('->')

    def Manhatten(self, node1, node2):
        state1 = node1.state
        state2 = node2.state
        h = 0
        for i in range(len(state1)):
            h += abs(state1[i]-state2[i])
        return h

    def orderOpen(self):
        fMin = self.open[-1].f
        for i in range(len(self.open)):
            if self.open[i].f <= fMin:
                fMin = self.open[i].f
                self.open = self.open[:i] + self.open[i+1:] + [self.open[i]]

    def search(self):
        while(True):
            if(len(self.open)):
                extandState=self.open[-1]
                spacIndex=extandState.state.index(0)
                flag=False
                if(extandState.degree>=self.MaxDegree):
                    node=self.open.pop()
                    self.close.append(node)
                    continue
                else:
                    for i in range(len(self.spce)):
                        if((i==0 and (spacIndex+self.spce[i])>=0) or
                        (i==1 and (spacIndex+self.spce[i])<len(extandState.state)-1)
                        or(i==2 and (spacIndex%self.length!=0 )) or
                        (i==3 and ((spacIndex+1)%self.length)!=0)):
                            state=self.copyArray(extandState.state)
                            #扩展状态
                            temp=state[spacIndex+self.spce[i]]
                            state[spacIndex+self.spce[i]]=0
                            state[spacIndex]=temp
                            nodeState=Node(extandState,state,extandState.degree+1)
                            targetNode = self.targetNode
                            nodeState.h = self.Manhatten(targetNode, nodeState)
                            nodeState.f = nodeState.degree + nodeState.h 
                            if(state==self.targetNode.state):
                                self.open.append(nodeState)
                                return True
                            elif( not self.isInTable(nodeState,self.close) and not self.isInTable(nodeState,self.open)):
                                self.open.append(nodeState)
                                flag=True
                            else:
                                continue
                    if(not flag):
                        self.open.pop()
                    else:
                        self.close.append(extandState)
                        self.open.remove(extandState)
                        self.orderOpen()

            else:
                return False
