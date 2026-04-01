from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
import gensim
from gensim.summarization import bm25
import numpy as np 
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import time
import os
import config
# 设置 SOCKS5 代理
# os.environ['http_proxy'] = 'socks5://127.0.0.1:10808'
# os.environ['https_proxy'] = 'socks5://127.0.0.1:10808'
device=torch.device('cuda:0')
X = list(range(config.mashup_num + config.API_num)) # MASHUP + api数量
mashup_id=X[config.API_num:]
#train_mashup_id, eval_mashup_id =[],[]
# MashupNumWithoutTest=4543
proportionOfNode=0.5 
stepOneMashupNum=100
maDict={}
description_text=[]
sumResNum=500
trussTopN=5

# def cross_validate(file_fold):
#     all_similarities = []
#     folds = []
#     with open(file_fold, 'r', encoding='utf-8') as f:
#         for line in f.readlines():
#             fold = eval(line.strip())  # 使用eval将类似[1,2,3]的字符串转为列表
#             folds.append(fold)
#
#     # 重新初始化数据集和模型
#     ems = initdsecDataset()  # 获取文本描述数据集
#
#     for fold_idx in range(10):  # 遍历所有折
#         print(f"Running fold {fold_idx + 1}...")
#
#         # 获取当前折的训练集和验证集
#         eval_mashup_id = folds[fold_idx]
#         train_folds = [folds[i] for i in range(10) if i != fold_idx]
#         train_mashup_id = [item for sublist in train_folds for item in sublist]  # 合并训练集
#
#         # 获取BM25分数
#         BM25_Scores = BM25Process(eval_mashup_id)
#
#         # 队列融合图特征重排序
#         RebuildNodeList, MashupScoreRes = ListRebuild2(BM25_Scores,eval_mashup_id)
#
#         # 构建步骤二数据集
#         leftDscps, rightData, rightApi, rightMashupScore = setStepTwoDataset(RebuildNodeList, MashupScoreRes,eval_mashup_id)
#
#         # 获取simcse相似度比较结果
#         simQA = simCsePro(eval_mashup_id, rightApi, ems, rightMashupScore)
#
#         # 将每个折的相似度结果保存下来
#         all_similarities.append(simQA)
#
#     return all_similarities

def cross_validate(file_fold):
    all_similarities = []
    folds = []
    with open(file_fold, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            fold = eval(line.strip())  # 使用eval将类似[1,2,3]的字符串转为列表
            folds.append(fold)

    # 重新初始化数据集和模型
    ems = initdsecDataset()  # 获取文本描述数据集

    for fold_idx in range(10):  # 遍历所有折
        if fold_idx < 5:
            continue
        print(f"Running fold {fold_idx + 1}...")

        # 获取当前折的训练集和验证集
        eval_mashup_id = folds[fold_idx]
        train_folds = [folds[i] for i in range(5) if i != fold_idx]
        train_mashup_id = [item for sublist in train_folds for item in sublist]  # 合并训练集

        # 获取BM25分数
        BM25_Scores = BM25Process(eval_mashup_id)

        # 队列融合图特征重排序
        RebuildNodeList, MashupScoreRes = ListRebuild2(BM25_Scores,eval_mashup_id)

        # 构建步骤二数据集
        leftDscps, rightData, rightApi, rightMashupScore = setStepTwoDataset(RebuildNodeList, MashupScoreRes,eval_mashup_id)

        # 获取simcse相似度比较结果
        simQA = simCsePro(eval_mashup_id, rightApi, ems, rightMashupScore)

        # 将每个折的相似度结果保存下来
        all_similarities.append(simQA)

    return all_similarities


def initdsecDataset():
    # 获取文本内容
    file_description = open('./data/data_description.txt', 'r', encoding='utf-8')
    for line in file_description.readlines():
        description_text.append(line.split(' =->= ')[1].replace("\n", ''))
    file_description.close()
    #获取MA关系
    edgeFile=open('./data/MA_Interactive.txt','r',encoding='utf-8')
    temp=[]
    nowIndex='1382'
    for line in edgeFile.readlines():
        Index=line.split()[0]
        value=line.split()[1]
        # print(Index,value)
        if(Index==nowIndex):
            temp.append(value)
        else:
            maDict[nowIndex]=temp
            temp=[]
            temp.append(value)
            nowIndex=Index
    maDict[nowIndex]=temp
    edgeFile.close()

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased").to(device)
    textIndex=0
    mainIndex=0
    textSim=[]
    emb=torch.Tensor().to(device)
    for textItem in description_text:
        textSim.append(textItem)
        textIndex=textIndex+1
        mainIndex=mainIndex+1
        if(textIndex==100):
            inputs = tokenizer(textSim, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to(device)
            textSim=[]
            emb=torch.cat([emb,embeddings],0)
            # print("正在获取第"+str(mainIndex)+"个词嵌入")
            textIndex=0
    inputs = tokenizer(textSim, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output.to(device)
    textSim=[]
    emb=torch.cat([emb,embeddings],0)
    print("已获取"+str(mainIndex)+"个句嵌入")
    textIndex=0
    print(len(emb))
    return emb

def BM25Process(eval_mashup_id):
    print("正在使用BM25计算文本相似度。。。")
    
    # print(description_text)
    all_description=[]
    for index in X:
        all_description.append(description_text[index].split())
    # print(all_description)

    testIndexItem=0
    Res=[]
    for testIndex in eval_mashup_id:
        bmDesArray=[]
        tempIndex=[]
        Res.append([])
        for textId in mashup_id:
            if textId==testIndex:
                continue
            else:
                tempIndex.append(textId)
                bmDesArray.append(all_description[textId])
        bm25Model = bm25.BM25(bmDesArray)
        # print(testIndex)
        scores=bm25Model.get_scores(all_description[testIndex])
        fullIndex=0
        for item in tempIndex:
            Res[testIndexItem].append([])
            Res[testIndexItem][fullIndex].append(item)
            Res[testIndexItem][fullIndex].append(scores[fullIndex])
            fullIndex=fullIndex+1
        testIndexItem=testIndexItem+1
    Index=0
    for Item in Res:
        Res[Index].sort(key=lambda x:x[0])
        Index=Index+1
    # print(testIndexItem)
    # print(Res[0])
    # print("BM25 List Length:"+str(len(Res[0])))
    return Res

def setStepTwoMashupId(sumScores):
    step2MashupList=[]
    for Index in range(0,len(sumScores)):
        step2MashupList.append([])
        for itemIndex in range(0,stepOneMashupNum):
            step2MashupList[Index].append(sumScores[Index][itemIndex])
    return step2MashupList

def getApiIdWithMashupId(mashupId):
    return maDict[str(mashupId)]

def setStepTwoDataset(step2MashupIdList,MashupScoreRes,eval_mashup_id):
    leftDscps=[]
    rightDscps=[]
    leftApi=[]
    rightApi=[]
    rightData=[]
    rightDataId=[]
    rightMashupScore=[]
    for Index in eval_mashup_id:
        leftDscps.append(description_text[Index])
        leftApiList=getApiIdWithMashupId(Index)
        leftApi.append(leftApiList)
    for ItemIndex in range(0,len(step2MashupIdList)):
        rightDscps.append([])
        rightApi.append([])
        rightMashupScore.append([])
        mashupIndex=0
        for item in step2MashupIdList[ItemIndex]:
            tempList=getApiIdWithMashupId(item)
            rightApi[ItemIndex]=rightApi[ItemIndex]+tempList
            for apiIndex in tempList:
                rightMashupScore[ItemIndex].append([MashupScoreRes[ItemIndex][mashupIndex][0],MashupScoreRes[ItemIndex][mashupIndex][1]])
                apiIndexNum=int(apiIndex)
                rightDscps[ItemIndex].append(description_text[apiIndexNum])
            mashupIndex=mashupIndex+1
    for Index in range(0,len(rightApi)):
        rightData.append([])
        rightDataId.append([])
        thisIndex=0
        for minIndex in range(0,len(rightApi[Index])):
            apiId=rightApi[Index][minIndex]
            apiDesc=rightDscps[Index][minIndex]
            rightDataId[Index].append(apiId)
            rightData[Index].append([])
            rightData[Index][thisIndex].append(apiId)
            rightData[Index][thisIndex].append(apiDesc)
            thisIndex=thisIndex+1
    return leftDscps,rightData,rightDataId,rightMashupScore

def simCsePro(eval_mashup_id,rightApi,embeddings,rightMashupScore):
    leftId=eval_mashup_id
    rightId=rightApi
    Res=[]
    IndexLeft=0
    import os

    folder_path = "./result"
    os.makedirs(folder_path, exist_ok=True)
    for leftItem in leftId:
        Res.append([])
        IndexRight=0
        texts = []
        #file=open("demo"+str(leftItem)+".txt",'w+')
        file = open("./result/demo" + str(leftItem) + ".txt", 'w+')
        #start =time.clock()
        start = time.perf_counter()

        texts.append(description_text[leftItem])
        # print(str(IndexLeft)+"/"+str(len(eval_mashup_id)))
        for rightItem in rightId[IndexLeft]:
            texts.append(description_text[int(rightItem)])        
        for rightItem in rightId[IndexLeft]:
            cosine_sim = 1 - cosine(embeddings[leftItem].cpu(), embeddings[int(rightItem)].cpu())
            Res[IndexLeft].append([])
            Res[IndexLeft][IndexRight].append(rightItem)
            Res[IndexLeft][IndexRight].append(cosine_sim)
            file.write(str(rightItem)+" "+str(cosine_sim)+' '+str(rightMashupScore[IndexLeft][IndexRight][0])+' '+str(rightMashupScore[IndexLeft][IndexRight][1])+'\n')
            IndexRight=IndexRight+1
        file.close()
        #end =time.clock()
        end = time.perf_counter()
        print('query '+str(IndexLeft+1)+"/"+str(len(eval_mashup_id)))
        IndexLeft=IndexLeft+1
    return Res

def ListRebuild2(BM25_Scores,eval_mashup_id):
    PATH = './data/mashup_node2vec_embeddings.txt'
    node_embeddings = gensim.models.KeyedVectors.load_word2vec_format(PATH, binary=False)
    Index=0
    Res=[]
    MashupScoreRes=[]
    for testIndexItem in BM25_Scores:
        testIndexItem.sort(key=lambda x:x[1],reverse=True)
        topNodeListBefore=testIndexItem[:trussTopN]
        topNodeList=[]
        for item in topNodeListBefore:
            topNodeList.append(item[0])
        topNodeSimList=[]
        revNodeList=[]
        scoreNodeList=[]
        itemIndexthis=0
        for item in testIndexItem[trussTopN:]:
            revNodeList.append([])
            revNodeList[itemIndexthis].append(int(item[0]))
            revNodeList[itemIndexthis].append(item[1])
            itemIndexthis=itemIndexthis+1
            scoreNodeList.append([])
        revNodeList.sort(key=lambda x:x[0],reverse=True)
        lsList=[]
        for item in revNodeList:
            lsList.append([item[0],item[1]])
        # 归一化
        npBm25=np.array(lsList)
        min_max_scaler = preprocessing.MinMaxScaler()
        npBm25_minMax = min_max_scaler.fit_transform(npBm25)
        lsList=npBm25_minMax.tolist()
        for item in range(0,len(revNodeList)):
            revNodeList[item][1]=lsList[item][1]
        # if(Index==0):
        #     print(revNodeList)
            
        # 归一化
        for topNItem in range(0,trussTopN):
            nodeSimRes=node_embeddings.similar_by_word(str(topNodeList[topNItem]),topn=config.mashup_num)
            # print(len(nodeSimRes))
            nodeTopIndex=0
            for nodeItem in nodeSimRes:
                if((int(nodeItem[0]) not in topNodeList) and int(nodeItem[0])!=eval_mashup_id[Index]):
                    scoreNodeList[topNItem].append([])
                    scoreNodeList[topNItem][nodeTopIndex].append(int(nodeItem[0]))
                    scoreNodeList[topNItem][nodeTopIndex].append(nodeItem[1])
                    nodeTopIndex=nodeTopIndex+1
            scoreNodeList[topNItem].sort(key=lambda x:x[0],reverse=True)
        nodeSimSum=[]

        # print('len(revNodeList)=',len(revNodeList))  # 检查 revNodeList 长度
        # print('len(scoreNodeList)=',len(scoreNodeList))  # 检查 scoreNodeList 长度

        for revLineIndex in range(0,len(revNodeList)):
            nodeSimSum.append(0)
        for topNItem in range(0,trussTopN):
            for revLineIndex in range(0,len(revNodeList)):
                try:
                    if (revNodeList[revLineIndex][0] == scoreNodeList[topNItem][revLineIndex][0]):
                        nodeSimSum[revLineIndex] = nodeSimSum[revLineIndex] + scoreNodeList[topNItem][revLineIndex][
                            1] / trussTopN
                except IndexError as e:
                    print("IndexError at revLineIndex =", revLineIndex, " topNItem =", topNItem)
                    print("revNodeList size:", len(revNodeList))
                    if revLineIndex < len(revNodeList):
                        print("revNodeList[revLineIndex] size:", len(revNodeList[revLineIndex]))
                    print("scoreNodeList size:", len(scoreNodeList))
                    if topNItem < len(scoreNodeList):
                        print("scoreNodeList[topNItem] size:", len(scoreNodeList[topNItem]))
                    raise e  # 保留原始错误抛出以定位行号
                if(revNodeList[revLineIndex][0]==scoreNodeList[topNItem][revLineIndex][0]):
                    nodeSimSum[revLineIndex]=nodeSimSum[revLineIndex]+scoreNodeList[topNItem][revLineIndex][1]/trussTopN
                # else:
                #     print("error!!!")
        # print(nodeSimSum)
        for revLineIndex in range(0,len(revNodeList)):
            revNodeList[revLineIndex][1]=revNodeList[revLineIndex][1]/2+nodeSimSum[revLineIndex]/2
        revNodeList.sort(key=lambda x:x[1],reverse=True)
        # print(len(revNodeList))
        # print(revNodeList)
        listResTemp=[]
        mashupScores=[]
        for item in topNodeList:
            mashupScores.append([item,1])
        for item in revNodeList[:sumResNum-trussTopN]:
            listResTemp.append(item[0])
            mashupScores.append([item[0],item[1]])
        Res.append(topNodeList+listResTemp)
        MashupScoreRes.append(mashupScores)
        # if(Index==0):
        #     print(Res)
        Index=Index+1
    return Res,MashupScoreRes



def main():
    all_similarities=cross_validate("./data/10_fold_numbered_new.txt")

if __name__ == '__main__':
    from memory_profiler import memory_usage

    mem_usage = memory_usage(main, interval=0.1)
    print(f"Peak memory: {max(mem_usage):.2f} MB")

    










