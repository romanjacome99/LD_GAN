from tensorflow.keras import layers


class UnetL(layers.Layer):
    def __init__(self, L=31):
        super(UnetL, self).__init__(name='unetl')
        # L = 31
        L = 64
        S = 16

        self.conv_11 = layers.Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_12 = layers.Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down_2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_21 = layers.Conv2D(2 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_22 = layers.Conv2D(2 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down_3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_31 = layers.Conv2D(3 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_32 = layers.Conv2D(3 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down_4 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv_41 = layers.Conv2D(4 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_42 = layers.Conv2D(4 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up_5 = layers.UpSampling2D(size=(2, 2))
        self.conv_50 = layers.Conv2D(3 * L, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.concat_5 = layers.Concatenate(axis=3)
        self.conv_51 = layers.Conv2D(3 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_52 = layers.Conv2D(3 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up_6 = layers.UpSampling2D(size=(2, 2))
        self.conv_60 = layers.Conv2D(2 * L, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.concat_6 = layers.Concatenate(axis=3)
        self.conv_61 = layers.Conv2D(2 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_62 = layers.Conv2D(2 * L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.up_7 = layers.UpSampling2D(size=(2, 2))
        self.conv_70 = layers.Conv2D(L, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.concat_7 = layers.Concatenate(axis=3)
        self.conv_71 = layers.Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_72 = layers.Conv2D(L, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        # self.conv_8 = layers.Conv2D(S, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv_8 = layers.Conv2D(S, 1)

    def call(self, inputs, **kwargs):

        x1 = self.conv_11(inputs)
        x1 = self.conv_12(x1)

        x2 = self.down_2(x1)
        x2 = self.conv_21(x2)
        x2 = self.conv_22(x2)

        x3 = self.down_3(x2)
        x3 = self.conv_31(x3)
        x3 = self.conv_32(x3)

        x4 = self.down_4(x3)
        x4 = self.conv_41(x4)
        x4 = self.conv_42(x4)

        # decoder

        x5 = self.up_5(x4)
        x5 = self.conv_50(x5)
        x5 = self.concat_5([x3, x5])
        x5 = self.conv_51(x5)
        x5 = self.conv_52(x5)

        x6 = self.up_6(x5)
        x6 = self.conv_60(x6)
        x6 = self.concat_6([x2, x6])
        x6 = self.conv_61(x6)
        x6 = self.conv_62(x6)

        x7 = self.up_7(x6)
        x7 = self.conv_70(x7)
        x7 = self.concat_7([x1, x7])
        x7 = self.conv_71(x7)
        x7 = self.conv_72(x7)

        x8 = self.conv_8(x7)

        return x8
