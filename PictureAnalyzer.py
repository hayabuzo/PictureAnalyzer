from skimage.io import imread, imshow, imsave
from skimage import img_as_float, img_as_ubyte
from skimage.filters import gaussian
from skimage.transform import resize

from numpy import dstack
import numpy as np
import os
import colorsys
import warnings
warnings.filterwarnings("ignore")

# Параметры

infolder = 'input/'    # папка для входных файлов
outfolder = 'output/'  # папка для выходных файлов
convert_mode = 'y'     # режим перевода в ЧБ: 'y' через YUV (быстрый), 'b' через HSB (медленный)

# создаем выходную директорию, если она еще не существует
if not os.path.exists(outfolder):
    os.mkdir(outfolder)

# Функции

# Функция очистки выходной папки
def clear_folder(folder):
    # получаем список файлов в папке
    dirlist = os.listdir(folder)
    # удаляем каждый файл
    if (len(dirlist)!=0):
        for i in range(len(dirlist)):
            os.remove(folder+dirlist[i])

# Функция создания карты насыщенности
def hsbmap(img,resx,chan):
    resy = int(img.shape[0]/img.shape[1]*resx)
    # создаем пиксельную матрицу
    ima = resize(img,(resy,resx),anti_aliasing=False)
    ima = img_as_ubyte(ima)
    # каждый пиксель переводим в HSB
    for y in range(ima.shape[0]):
        for x in range(ima.shape[1]):
            # создаем дополнительный массив для масштабирования B
            a = np.array( colorsys.rgb_to_hsv(ima[y,x,0],ima[y,x,1],ima[y,x,2]) )
            # приводим B к float размерам H и S
            a[2] = a[2]/255
            # пререводим матрицу обратно в целые
            a = img_as_ubyte(a)
            # присваиваем изображению значения матрицы
            ima[y,x] = a
    # выводим нужный канал
    return ima[:,:,chan]

def average(img):
    avcolor = np.array( [0 for i in range(img.shape[2])] , dtype=np.uint8 )
    for i in range(img.shape[2]):
        avcolor[i] = img[:,:,i].sum() / (img.size/img.shape[2])
    #avcolor=dstack((avcolor[0],avcolor[1],avcolor[2]))
    return avcolor

# Функция расчета композиции        
def xshift(img,preview,bwmode):
    
    # определяем координаты середины кадра
    ys = img.shape[0]//2
    xs = img.shape[1]//2
    # создаем ч/б маску 
    if bwmode=='y':
        imy = 0.2126*img[:,:,0]+0.7152*img[:,:,1]+0.0722*img[:,:,2]
    if bwmode=='b':
        imy = hsbmap(img,80,2)
        imy = resize(imy,(img.shape[0],img.shape[1]),anti_aliasing=False)
        imy = img_as_ubyte(imy)
    # автоконтраст с отброшенными %
    k = round(imy.size * 0.01) # процент отброса
    imy_sorted = np.sort(imy.ravel())
    xmin=imy_sorted[k] 
    xmax=imy_sorted[-k]
    imyc = (imy-xmin)*255/(xmax-xmin)
    imyc = np.clip(imyc,0,255)
    if ((imyc.max()-imyc.min())>= 10): # проверка на однотонность
        imy = imyc
    imy = np.array(imy, dtype=np.uint8)
    # считаем число пикселей в половине кадра
    ps = imy.size//2
    # размываем маску для отбрасывания шумов
    kblur = 150 # радиус размытия
    imyb =  gaussian(imy,(xs+ys)/kblur)
    imyb = img_as_ubyte(np.clip(imyb,0,1))
    # создаем темный градиент для плеча массы
    grad = np.array( [[abs(255-j) for j in range(512)] for i in range(255)] )
    grad = np.array(grad, dtype=np.uint8)
    grad = resize(grad,(img.shape[0],img.shape[1]),anti_aliasing=False)
    
    # определяем баланс светов/теней
    
    # сумма света всех пикселей
    Wforce = imyb.sum()
    # сумма тени всех пикселей
    Bforce = imyb.size*255-Wforce
    # отношение света к тени
    BW = Wforce/Bforce
    # создаем корректирующий коэффициент из отношения: баланс света/тени 2 к 1 равняется 1 значения коэффициента
    BW = np.log2(BW)
    # ограничиваем коэффициент единицей
    BW = np.clip(BW,-1,1)
    # отношение света к площади кадра
    wf = Wforce/(imyb.size*255)
    if (BW >= 0):
        # кадр светлый, вес зависит от темных пятен
        mtype = '(B)'
        # выделяем темные пятна, добавляем плечо градиента
        imd = (255-imyb) * grad
    else:
        # кадр темный, вес зависит от светлых пятен
        mtype = '(W)'
        # выделяем светлые пятна, добавляем плечо градиента
        imd = imyb * grad
    # переводим карту световых пятен в целые числа
    imd = np.array(imd, dtype=np.uint8)
    # расчет перевеса световых пятен в левой и правой части кадра
    ML = int(imd[:,:xs].sum() +1) # прибавляем единицу для защиты от деления на ноль
    MR = int(imd[:,xs:].sum() +1)
    # расчет общего перевеса с учетом коэффициента
    MLR = int( (MR-ML)/(MR+ML) * 100 * (abs(BW)) )
    
    # определяем баланс контуров объектов
        
    # повторно размываем маску для отделения контуров  
    imyb1 = img_as_ubyte( gaussian( np.clip(imyb ,50,205) ,(xs+ys)/(kblur*2)) )
    imyb2 = img_as_ubyte( gaussian( np.clip(imyb1,50,205) ,(xs+ys)/(kblur*2)) )
    # отделяем контуры
    ims = imyb2-imyb1
    # добавляем плечо градиента
    ims = ims * grad
    ims = np.array(ims, dtype=np.uint8)
    # расчет перевеса контуров в левой и правой части
    CL = int(ims[:,:xs].sum()+1)
    CR = int(ims[:,xs:].sum()+1)
    # расчет общего перевеса
    CLR = int( (CR-CL)/(CR+CL) * 100 )
    # процент контуров в изображении
    cf = (CR+CL)/(ps*255)
    
    # определяем баланс насыщенности
    
    # создаем карту насыщенности
    smap = hsbmap(img,40,1)
    smap = resize(smap,(img.shape[0],img.shape[1]),anti_aliasing=False)*imyb
    # добавляем плечо градиента
    smap = smap * grad
    smap = np.array(smap, dtype=np.uint8)
    # расчет перевеса насыщенности в левой и правой части
    SL = int( smap[ :, :smap.shape[1]//2 ].sum() + 1 ) # защита от деления на 0
    SR = int( smap[ :, smap.shape[1]//2: ].sum() + 1 )
    # расчет общего перевеса
    SLR = int( (SR-SL)/(SR+SL) * 100 )
    # процент насыщенности изображения
    sf = (SL+SR)/(imyb.size*255)
    
    # общий вектор перевесов
    TX = (CLR+MLR+SLR)
    # модуль перевесов
    PX = abs(CLR)+abs(MLR)+abs(SLR)
    
    # объединяем маски для создания превью
    imout = np.vstack(( np.hstack((ims,imyb)),np.hstack((smap,imd)) ))
    
    if (preview==1):
        imshow(imout)
        print()
        print()
        print('total =',TX)    
        print('detail =',CLR, 'mass =',MLR, mtype, 'saturation =',SLR)
    
    return TX, PX, CLR, MLR, SLR, mtype, wf, cf, sf, imout

# Функция сортировки по равновесию
def xsort(folder,filt,filp,vis,chan):    
    # получаем список файлов в папке
    flist = os.listdir(folder)
    # для каждого файла производим обработку
    for i in range(len(flist)):
        # считываем файл
        img = imread(folder+flist[i])

        # уменьшение изображения для визуализации показателей
        if (vis==1):
            # коэффициент формата изображения, отношение ширины и высоты
            ky = img.shape[1]/img.shape[0]
            # ширина выходного изображения
            lx = 400
            # высота выходного изображения
            ly = int(lx/ky)
            # координаты середины кадра
            lxp = lx//2
            lyp = ly//2
            # уменьшаем изображение для наложения графиков
            img = resize(img,(ly,lx))
            img = img_as_ubyte(img)  

        # расчитываем показатели анализа изображения
        tx, px, clr, mlr, slr, mtype, wf, cf, sf, imout = xshift(img, preview=0, bwmode=convert_mode)
        
        # создание визуализаций
        if (vis==1):      
            # половина ширины вертикальной линии с контуром
            l2 = 2
            # половина ширины вертикальной линии без контура
            l1 = 1
            # половина высоты горизонтальных линий с контуром
            w2 = 4
            # половина высоты горизонтальных линий без контура
            w1 = 3
            
            # положение показателя TX по высоте
            posy_tx = lyp//2
            # цвет показателя ТХ - ГОЛУБОЙ (общий вектор равновесия)
            col_tx = [145,255,255]

            # положение показателя MLR по высоте 
            posy_mlr = lyp//2+lyp//4
            # цвет показателя MLR в зависимости от расчета темных/светлых пятен
            if mtype==('(B)'):
                col_mlr = 0
            else:
                col_mlr = 255

            # положение показателя CLR по высоте
            posy_clr = lyp
            # цвет показателя CLR - ЗЕЛЕНЫЙ (количество и равновесие контуров, деталей изображения)
            col_clr = [145,255,145]

            # положение показателя SLR по высоте
            posy_slr = lyp+lyp//4
            # цвет показателя SLR - КРАСНЫЙ (количество и равновесие насыщенности в изображении)
            col_slr = [255,145,145]  

            # масштабируем показатели до ширины изображения
            wf = int(lx*wf)
            cf = int(lx*cf)
            sf = int(lx*sf)
            # положение показателей по высоте
            posy_wf = ly-(w2*6)+w1
            posy_cf = ly-(w2*4)+w1
            posy_sf = ly-(w2*2)+w1
            # цвет показателя WF - БЕЛЫЙ (количество и равновесие светлых пикселей в изображении)
            col_wf = [255,255,255]  

            # построение показателей для положительных/отрицательных значений
            if tx>0:
                img[ posy_tx-w2:posy_tx+w2, lxp:lxp+tx+w2 ] = 0
                img[ posy_tx-w1:posy_tx+w1, lxp:lxp+tx+w1 ] = col_tx
            else:
                img[ posy_tx-w2:posy_tx+w2, lxp+tx-w2:lxp ] = 0
                img[ posy_tx-w1:posy_tx+w1, lxp+tx-w1:lxp ] = col_tx
                
            if mlr>0:
                img[ posy_mlr-w2:posy_mlr+w2, lxp:lxp+mlr+w2 ] = 255-col_mlr
                img[ posy_mlr-w1:posy_mlr+w1, lxp:lxp+mlr+w1 ] = col_mlr
            else:
                img[ posy_mlr-w2:posy_mlr+w2, lxp+mlr-w2:lxp ] = 255-col_mlr
                img[ posy_mlr-w1:posy_mlr+w1, lxp+mlr-w1:lxp ] = col_mlr
                
            if clr>0:
                img[ posy_clr-w2:posy_clr+w2, lxp:lxp+clr+w2 ] = 0
                img[ posy_clr-w1:posy_clr+w1, lxp:lxp+clr+w1 ] = col_clr
            else:
                img[ posy_clr-w2:posy_clr+w2, lxp+clr-w2:lxp ] = 0
                img[ posy_clr-w1:posy_clr+w1, lxp+clr-w1:lxp ] = col_clr
                
            if slr>0:
                img[ posy_slr-w2:posy_slr+w2, lxp:lxp+slr+w2 ] = 0
                img[ posy_slr-w1:posy_slr+w1, lxp:lxp+slr+w1 ] = col_slr
            else:
                img[ posy_slr-w2:posy_slr+w2, lxp+slr-w2:lxp ] = 0
                img[ posy_slr-w1:posy_slr+w1, lxp+slr-w1:lxp ] = col_slr

            # построение вертикальной линии    
            img[:,lxp-l2:lxp+l2] = 0
            img[:,lxp-l1:lxp+l1] = 255

            # построение процентных показателей
            img[ posy_wf-w2:ly, 0:lx ] = 0   
            img[ posy_wf-w1:posy_wf+w1, 0:wf+w1 ] = col_wf
            img[ posy_cf-w1:posy_cf+w1, 0:cf+w1 ] = col_clr
            img[ posy_sf-w1:posy_sf+w1, 0:sf+w1 ] = col_slr
        
        # масштабирование показателей для избавления от отрицательных значений       
        tx2 = 100+(mlr+clr+slr)//3
        px2 = px
        cb2 = 100+clr
        mb2 = 100+mlr
        sb2 = 100+slr
        
        # расчет HSB
        aim = average(img)
        aim = colorsys.rgb_to_hsv(aim[0],aim[1],aim[2])
        ahue = int(aim[0]*255)
        asat = int(aim[1]*255)
        aval = aim[2]
        ahb = aval//10*1000+ahue
        
        # фильтрация изображений по равновесию и разбросу
        if (abs(tx)<filt*2) and (px2>filp):
            if (chan==1):
                # масштабирование расчетных каналов до размера изображения для склейки
                imout = resize(imout,(img.shape[0],img.shape[1]),anti_aliasing=False)
                imout = img_as_ubyte(imout)
                imout = dstack((imout,imout,imout))
                # склейка изображения с расчетными каналами
                img = np.hstack((img,imout))
                
            # сохранение изображения
            
            imname = outfolder # путь для сохранения
            
            # выбор параметра для сортировки
            
            imname = imname + 'wf='+str(int(wf*100))+' '   # процент освещенности (БЕЛЫЙ)
            imname = imname + 'cf='+str(int(cf*100))+' '   # процент детализации (ЗЕЛЕНЫЙ)
            imname = imname + 'sf='+str(int(sf*100))+' '   # процент насыщенности (КРАСНЫЙ)
            
            #imname = imname + 'T='+str(tx2)+' '            # сумма векторов равновесия (ГОЛУБОЙ)
            #imname = imname + 'P='+str(px2)+' '            # сумма модулей векторов равновесия
            
            #imname = imname + 'M='+str(mb2)+mtype+' '      # отклонение по световым/теневым пятнам
            #imname = imname + 'C='+str(cb2)+' '            # отклонение по детализации
            #imname = imname + 'S='+str(sb2)+' '            # отклонение по насыщенности
            
            imname = imname + 'HUE='+str(ahue)+' '         # цветовой оттенок 
            imname = imname + 'SAT='+str(asat)+' '         # насыщенность
            imname = imname + 'VAL='+str(aval)+' '         # яркость 
            #imname = imname + 'AHB='+str(ahb)+' '      # цветовой оттенок с учетом яркости
            
            imname = imname + '# '+flist[i] # имя исходного файла
            
            imsave(imname,img)
            
        # вывод строки состояния    
        print("\r",str(i+1)+'/'+str(len(flist))+' - READY -', flist[i], end="")  



# Программа

# очищаем выходную папку
clear_folder(outfolder)

# сортируем изображения из входной папки в нужном режиме
# параметры: директория, фильтр по отклонению равновесия, фильтр по модулю векторов, визуализация показателей в виде баров, визуализация расчетных каналов

xsort(infolder, 10000000, -5000000, vis=1, chan=0)

print()
print('COMPLETED')
