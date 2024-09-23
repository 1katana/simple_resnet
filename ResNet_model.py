import tensorflow as tf
import tensorflow as tf
from keras import layers, Model, Input

class ResidualBlock(layers.Layer):
    def __init__(self,number, filters, kernel_size=3, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.list_residual=[]

        for i in range(1,number+1):
            print(self.kernel_size)
            self.list_residual.append([
                 
                layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1,padding="same"),
                layers.BatchNormalization(),
                layers.ReLU()
            ])

        self.maxpool=layers.MaxPool2D(pool_size=[2,2])       
            

        self.shortcut = layers.Conv2D(filters=self.filters, kernel_size=1, strides=2)
        self.shortcut_norm = layers.BatchNormalization()

        # Выходной ReLU
        self.ReLU_output = layers.ReLU()
        
    def call(self, x: tf.Tensor):
        shortcut = self.shortcut(x)
        shortcut = self.shortcut_norm(shortcut)
        
        for layer in self.list_residual:
            
            for i in range(2):
                x=layer[i](x)
        x=self.maxpool(x)

        # Добавляем shortcut к выходу основного блока
        x = layers.add([x, shortcut])
        x = self.ReLU_output(x)

        return x
    
    def compute_output_shape(self, input_shape):
        shape = self.maxpool.compute_output_shape(input_shape)
        return shape

class ResNet50(Model):
    
    def __init__(self, input_shape, num_classes: int, num_block_list):
        super().__init__()
        
        self.num_block_list = num_block_list
        self.filters = 16  # Начальное количество фильтров
        
        # Входной слой
        self.input_layer = Input(input_shape)
        
        # Первый свёрточный слой
        self.Conv1 = layers.Conv2D(filters=self.filters, kernel_size=5, strides=1, padding="same", activation="relu")
        self.norm1 = layers.BatchNormalization()

        # Создание residual-блоков
        self.residual_blocks = []
        for i, num_block in enumerate(self.num_block_list):
            self.residual_blocks.append(ResidualBlock(num_block,self.filters))
            self.filters *= 2  # Увеличиваем количество фильтров в каждом блоке
        
        # Global Average Pooling и выходной полносвязный слой
        self.global_pooling = layers.GlobalAveragePooling2D()
        self.outputs = layers.Dense(num_classes, activation="softmax")
    
    def summary(self, **kwargs):
        return super().summary(**kwargs)
    
    def call(self, x: tf.Tensor):
        # Проход через начальные слои
        x = self.Conv1(x)
        x = self.norm1(x)

        # Проход через residual-блоки
        for block in self.residual_blocks:
            x = block(x)
        
        # Global Average Pooling и финальный Dense слой
        x = self.global_pooling(x)
        x = self.outputs(x)
        
        return x