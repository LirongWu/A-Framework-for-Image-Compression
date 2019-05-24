#-*- coding: UTF-8 -*-
import os
import math
import numpy as np
import scipy.misc
import tensorflow as tf
from datetime import datetime

from avatar import Avatar
from skimage.measure import compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class AvatarModel:

    def __init__(self):
        self.avatar = Avatar()
        # 真实图片shape (height, width, depth)
        self.img_shape = self.avatar.img_shape
        # 一个batch的图片向量shape (batch, height, width, depth)
        self.batch_shape = self.avatar.batch_shape
        # 一个batch包含图片数量
        self.batch_size = self.avatar.batch_size
        # batch的总数量
        self.chunk_size = self.avatar.chunk_size
        # 迭代次数
        self.epoch_size = 256
        # 学习率
        self.learning_rate = 2e-4
        # 优化指数衰减率
        self.beta1 = 0.5
        # channal
        self.channal = 8

    def lrelu(x):
        return tf.maximum(x, 0.2*x)

    def batch_normalizer(x, train=True):
        return tf.layers.batch_normalization(x, momentum=0.9, epsilon=1e-4, scale=True, renorm=False, training=train, axis=-1)

    @staticmethod
    def conv_block(self, x, filters, kernel_size, strides, padding, actv=tf.nn.relu, train=True, name='conv_block'):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None)
        x = self.batch_normalizer(x, train=train)
        x = actv(x)
        return x

    @staticmethod
    def residual_block(self, x, filters, kernel_size, strides, actv=tf.nn.relu, train=True, name='residual_block'):
        identity_map = x

        res = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID' ,activation=None)
        res = self.batch_normalizer(res, train=train)
        res = tf.nn.relu(res)

        res = tf.pad(res, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID' ,activation=None)
        res = self.batch_normalizer(res, train=train)

        out = tf.add(res, identity_map)
        return out

    @staticmethod
    def upsample_block(self, x, filters, kernel_size, strides, padding, actv=tf.nn.relu, train=True, name='upsample_block'):
        x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None)
        x = self.batch_normalizer(x, train=train)
        x = actv(x)
        return x

    def encoder(self, input_imgs, train=True):

        layer_1 = self.conv_block(self, x=input_imgs, filters=64, kernel_size=3, strides=1, padding='SAME', train=train, name='encoder_1') 
        layer_2 = self.conv_block(self, x=layer_1, filters=64, kernel_size=3, strides=2, padding='SAME', train=train, name='encoder_2') 
        layer_3 = self.conv_block(self, x=layer_2, filters=128, kernel_size=3, strides=1, padding='SAME', train=train, name='encoder_3') 
        layer_4 = self.conv_block(self, x=layer_3, filters=128, kernel_size=3, strides=2, padding='SAME', train=train, name='encoder_4') 
        layer_5 = self.conv_block(self, x=layer_4, filters=256, kernel_size=3, strides=1, padding='SAME', train=train, name='encoder_5') 
        layer_6 = self.conv_block(self, x=layer_5, filters=256, kernel_size=3, strides=2, padding='SAME', train=train, name='encoder_6') 

        layer_7 = self.residual_block(self, x=layer_6, filters=256, kernel_size=3, strides=1, train=train, name='encoder_7')
        layer_8 = self.residual_block(self, x=layer_7, filters=256, kernel_size=3, strides=1, train=train, name='encoder_8')
        layer_9 = self.residual_block(self, x=layer_8, filters=256, kernel_size=3, strides=1, train=train, name='encoder_9')

        layer_10 = self.conv_block(self, x=layer_9, filters=self.channal, kernel_size=3, strides=1, padding='SAME', train=train, name='encoder_10') 

        return layer_10


    def quantizer(slef, imgs_encoder):
        centers = tf.cast(tf.range(2), tf.float32)
        w_stack = tf.stack([imgs_encoder for _ in range(2)], axis=-1)
        w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
        smx = tf.nn.softmax(-1.0 * tf.abs(w_stack - centers), dim=-1)
        w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)  
        q = tf.stop_gradient(w_hard - w_soft) + w_soft
        return q

    def generator(self, q, train=True):
        out = self.conv_block(self, x=q, filters=64, kernel_size=3, strides=1, padding='SAME', train=train, name='generator_1') 
        out = self.conv_block(self, x=out, filters=128, kernel_size=3, strides=1, padding='SAME', train=train, name='generator_2') 
        out = self.conv_block(self, x=out, filters=256, kernel_size=3, strides=1, padding='SAME', train=train, name='generator_3') 


        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_4')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_5')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_6')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_7')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_8')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_9')
        out = self.residual_block(self, x=out, filters=256, kernel_size=3, strides=1, train=train, name='generator_10')
        
        out = self.upsample_block(self, x=out, filters=256, kernel_size=3, strides=2, padding='SAME', train=train, name='generator_11') 
        out = self.upsample_block(self, x=out, filters=128, kernel_size=3, strides=2, padding='SAME', train=train, name='generator_12') 
        out = self.upsample_block(self, x=out, filters=64, kernel_size=3, strides=2, padding='SAME', train=train, name='generator_13') 

        out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out = tf.layers.conv2d(out, filters=3, kernel_size=3, strides=1, padding='VALID', name='generator_14')
        out = tf.nn.tanh(out)
        
        return out


    def discriminator(self, x, train=True):

        c1 = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='SAME', activation=self.lrelu)
        c2 = self.conv_block(self, x=c1, filters=128, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu)
        c3 = self.conv_block(self, x=c2, filters=256, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu)
        c4 = self.conv_block(self, x=c3, filters=512, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu)
        out = tf.layers.conv2d(c4, filters=1, kernel_size=3, strides=1, padding='SAME')
        # out = tf.sigmoid(out)
        return out

    @staticmethod
    def loss_graph(real, fake, real_imgs, generator_imgs):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        D_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        D_loss = D_loss_real + D_loss_gen
        G_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        G_loss_similarity = tf.losses.mean_squared_error(real_imgs, generator_imgs)  
        G_loss = G_loss_gen +  12 * G_loss_similarity
        return D_loss, G_loss


    def train(self):

        real_imgs = tf.placeholder(tf.float32, self.batch_shape, name='real_images')
        # 生成器
        with tf.variable_scope('generator'):

            encoder_imgs = self.encoder(real_imgs)
            quantizer_imgs = self.quantizer(encoder_imgs) 
            generator_imgs = self.generator(quantizer_imgs)

        # 判别器
        with tf.variable_scope('discriminator'):
            D_x = self.discriminator(real_imgs)
            D_Gx = self.discriminator(generator_imgs)

        # 损失
        D_loss, G_loss = self.loss_graph(D_x, D_Gx, real_imgs, generator_imgs)
        # 优化
        train_vars = tf.trainable_variables()
        # 生成器变量
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        # 判别器变量
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]

        # 生成器与判别器作为两个网络需要分别优化
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(G_loss, var_list=gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(D_loss, var_list=dis_vars)

        # 开始训练
        saver = tf.train.Saver()
        step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "/data/wulirong/GAN/model/avatar.model")
            for epoch in range(self.epoch_size):
                try:
                    batches = self.avatar.batches()
                    for batch_imgs in batches:
                        _ = sess.run(dis_optimizer, feed_dict={real_imgs: batch_imgs})
                        _ = sess.run(gen_optimizer, feed_dict={real_imgs: batch_imgs})
                        _ = sess.run(gen_optimizer, feed_dict={real_imgs: batch_imgs})
                        loss_d = sess.run(D_loss, feed_dict={real_imgs: batch_imgs})
                        loss_g = sess.run(G_loss, feed_dict={real_imgs: batch_imgs})
                        step += 1
                        print(datetime.now().strftime('%c'), epoch, step, 'D_loss:', loss_d, 'G_loss:', loss_g)

                    step = 0

                except:
                	model_path = os.getcwd() + os.sep + "model/avatar.model"
                	saver.save(sess, model_path)

            model_path = os.getcwd() + os.sep + "avatar.model"
            saver.save(sess, model_path)


    def gen(self):
        # 压缩图片
        real_imgs = tf.placeholder(tf.float32, self.batch_shape, name='test_images')
        with tf.variable_scope('generator'):
            encoder_imgs = self.encoder(real_imgs)
            quantizer_imgs = self.quantizer(encoder_imgs)       
            generator_imgs = self.generator(quantizer_imgs)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            batches = self.avatar.batches()
            for test_imgs in batches:
                compress_result = sess.run(generator_imgs, feed_dict={real_imgs: test_imgs})
                for num in range(len(compress_result)):
                    scipy.misc.imsave('results'+os.sep+str(num)+'.png', compress_result[num])