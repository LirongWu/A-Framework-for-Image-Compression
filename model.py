#-*- coding: UTF-8 -*-
import os
import cv2
import random
import numpy as np
import scipy.misc
import tensorflow as tf
from datetime import datetime
from skimage.measure import compare_ssim

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,7"

class Model:

    def __init__(self):

        self.img_shape = (240, 320, 3)
        self.block_h = 24
        self.block_w = 32
        self.block_num = 10
        self.video_size = 25

        self.epoch_size = 32
        self.learning_rate = 2e-4
        self.beta1 = 0.5

    def lrelu(self, x):

        return tf.maximum(x, 0.2*x)

    def conv_block(self, x, filters, kernel_size, strides, padding, actv=tf.nn.relu, train=True, name='conv_block'):
        x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'conv')
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-4, scale=True, renorm=False, training=train, axis=-1, name=name+'batch')
        x = actv(x)
        return x

    def upsample_block(self, x, filters, kernel_size, strides, padding, actv=tf.nn.relu, train=True, name='upsample_block'):
        x = tf.layers.conv2d_transpose(x, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'conv')
        x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-4, scale=True, renorm=False, training=train, axis=-1, name=name+'batch')
        x = actv(x)
        return x

    def residual_block(self, x, filters, kernel_size, strides, actv=tf.nn.relu, train=True, name='residual_block'):
        identity_map = x

        res = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID', name=name+'conv1')
        res = tf.layers.batch_normalization(res, momentum=0.99, epsilon=1e-4, scale=True, renorm=False, training=train, axis=-1, name=name+'batch1')
        res = tf.nn.relu(res)

        res = tf.pad(res, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        res = tf.layers.conv2d(res, filters=filters, kernel_size=kernel_size, strides=strides, padding='VALID', name=name+'conv2')
        res = tf.layers.batch_normalization(res, momentum=0.99, epsilon=1e-4, scale=True, renorm=False, training=train, axis=-1, name=name+'batch2')

        out = tf.add(res, identity_map)

        return out

    def quantizer(slef, x):
        centers = tf.cast(tf.range(4), tf.float32)
        w_stack = tf.stack([x for _ in range(4)], axis=-1)
        w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
        smx = tf.nn.softmax(-1.0 * tf.abs(w_stack - centers), dim=-1)
        w_soft = tf.einsum('ijklm,m->ijkl', smx, centers)  
        w_bar = tf.stop_gradient(w_hard - w_soft) + w_soft
        return w_bar

    def foreground_encoder(self, x, train=True):
        x = tf.reshape(x, [1, self.block_h, self.block_w, 3])
        layer_1 = self.conv_block(x=x, filters=64, kernel_size=5, strides=2, padding='SAME', train=train, name='crop_encoder_1') 
        layer_2 = self.residual_block(x=layer_1, filters=64, kernel_size=3, strides=1, train=train, name='crop_encoder_2')

        layer_3 = self.conv_block(x=layer_2, filters=128, kernel_size=4, strides=2, padding='SAME', train=train, name='crop_encoder_3') 
        layer_4 = self.residual_block(x=layer_3, filters=128, kernel_size=3, strides=1, train=train, name='crop_encoder_4')


        layer_5 = self.conv_block(x=layer_4, filters=256, kernel_size=4, strides=2, padding='SAME', train=train, name='crop_encoder_5') 
        layer_6 = self.residual_block(x=layer_5, filters=256, kernel_size=3, strides=1, train=train, name='crop_encoder_6')
        layer_7 = self.residual_block(x=layer_6, filters=256, kernel_size=3, strides=1, train=train, name='crop_encoder_7')
        layer_8 = self.residual_block(x=layer_7, filters=256, kernel_size=3, strides=1, train=train, name='crop_encoder_8')

        layer_8 = tf.pad(layer_8, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        layer_9 = self.conv_block(x=layer_8, filters=16, kernel_size=3, strides=1, padding='VALID', train=train, name='crop_encoder_9')
        return layer_9

    def foreground_decoder(self, x, train=True):
        out = self.conv_block(x=x, filters=128, kernel_size=3, strides=1, padding='SAME', train=train, name='crop_decoder_1') 
        out = self.conv_block(x=out, filters=256, kernel_size=3, strides=1, padding='SAME', train=train, name='crop_decoder_2') 

        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_3')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_4')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_5')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_6')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_7')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='crop_decoder_8')

        out = self.upsample_block(x=out, filters=256, kernel_size=4, strides=2, padding='SAME', train=train, name='crop_decoder_9') 
        out = self.upsample_block(x=out, filters=128, kernel_size=4, strides=2, padding='SAME', train=train, name='crop_decoder_10') 
        out = self.upsample_block(x=out, filters=64, kernel_size=5, strides=2, padding='SAME', train=train, name='crop_decoder_11')

        out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out = tf.layers.conv2d(out, filters=3, kernel_size=3, strides=1, padding='VALID', name='crop_decoder_12')
        out = tf.nn.tanh(out)
        out = tf.reshape(out, [self.block_h, self.block_w, 3])
        return out

    def foreground_codec(self, real_img):

        x = random.randint(0, self.block_num - 1)
        y = random.randint(0, self.block_num - 1)
        crop_img = real_img[y*self.block_h:(y+1)*self.block_h, x*self.block_w:(x+1)*self.block_w, :]

        crop_code = self.foreground_encoder(crop_img)
        quantize = self.quantizer(crop_code)
        res_img = self.foreground_decoder(quantize)

        return crop_img, res_img, quantize

    def quality_enhance(self, x, train=True):
        out = tf.reshape(x, [1, self.block_h*self.block_num, self.block_w*self.block_num, 3])

        out = self.conv_block(x=out, filters=64, kernel_size=5, strides=2, padding='SAME', train=train, name='enhance_1') 
        out = self.conv_block(x=out, filters=128, kernel_size=4, strides=2, padding='SAME', train=train, name='enhance_2')
        out = self.conv_block(x=out, filters=256, kernel_size=4, strides=2, padding='SAME', train=train, name='enhance_3')

        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='enhance_4')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='enhance_5')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='enhance_6')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='enhance_7')
        out = self.residual_block(x=out, filters=256, kernel_size=3, strides=1, train=train, name='enhance_8')

        out = self.upsample_block(x=out, filters=256, kernel_size=4, strides=2, padding='SAME', train=train, name='enhance_9') 
        out = self.upsample_block(x=out, filters=128, kernel_size=4, strides=2, padding='SAME', train=train, name='enhance_10') 
        out = self.upsample_block(x=out, filters=64, kernel_size=5, strides=2, padding='SAME', train=train, name='enhance_11')

        out = tf.pad(out, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out = tf.layers.conv2d(out, filters=3, kernel_size=3, strides=1, padding='VALID', name='enhance_12')
        out = tf.nn.tanh(out)
        out = tf.reshape(out, [self.block_h*self.block_num, self.block_w*self.block_num, 3])
        return out

    def f1(self, real_img, i, j):
        out = self.foreground_encoder(real_img[i*self.block_h:(i+1)*self.block_h, j*self.block_w:(j+1)*self.block_w, :])
        return out
        
    def f2(self, bg_img, codes, i, j):
        quantize = self.quantizer(codes)
        res_img = self.foreground_decoder(quantize)
        ans = tf.subtract(res_img, res_img + 1.0)
        ans = tf.pad(ans, [[i*self.block_h, (self.block_num-1-i)*self.block_h], [j*self.block_w, (self.block_num-1-j)*self.block_w], [0, 0]], 'CONSTANT')
        ans = tf.add(ans, 1.0)
        res_img = tf.pad(res_img, [[i*self.block_h, (self.block_num-1-i)*self.block_h], [j*self.block_w, (self.block_num-1-j)*self.block_w], [0, 0]], 'CONSTANT')
        result = tf.add(res_img, tf.multiply(ans, bg_img))
        return result

    def f3(self, bg_img, real_img, i, j):
        codes = self.foreground_encoder(real_img[i*self.block_h:(i+1)*self.block_h, j*self.block_w:(j+1)*self.block_w, :])
        quantize = self.quantizer(codes)
        res_img = self.foreground_decoder(quantize)
        ans = tf.subtract(res_img, res_img + 1.0)
        ans = tf.pad(ans, [[i*self.block_h, (self.block_num-1-i)*self.block_h], [j*self.block_w, (self.block_num-1-j)*self.block_w], [0, 0]], 'CONSTANT')
        ans = tf.add(ans, 1.0)
        res_img = tf.pad(res_img, [[i*self.block_h, (self.block_num-1-i)*self.block_h], [j*self.block_w, (self.block_num-1-j)*self.block_w], [0, 0]], 'CONSTANT')
        result = tf.add(res_img, tf.multiply(ans, bg_img))
        return result

    def composite(self, bg_img, real_img, fg_img):
        num = 0
        for i in range(self.block_num):
            for j in range(self.block_num):
                average = tf.reduce_mean(fg_img[i*self.block_h:(i+1)*self.block_h, j*self.block_w:(j+1)*self.block_w])
                # code = tf.cond(tf.greater(average, -0.92), lambda: self.f1(real_img, i, j), lambda: tf.zeros([1, self.block_h/8, self.block_w/8, 16]))
                # bg_img = tf.cond(tf.greater(average, -0.92), lambda: self.f2(bg_img, code, i, j), lambda: bg_img)
                bg_img = tf.cond(tf.greater(average, -0.90), lambda: self.f3(bg_img, real_img, i, j), lambda: bg_img)
                num = tf.cond(tf.greater(average, -0.90), lambda: num + 1, lambda: num)
        return bg_img, num

    def discriminator(self, x, train=True):
        x = tf.reshape(x, [1, self.block_h*self.block_num, self.block_w*self.block_num, 3])
        d1 = tf.layers.conv2d(x, filters=64, kernel_size=3, strides=2, padding='SAME', activation=self.lrelu, name='dis_1')
        d2 = self.conv_block(x=d1, filters=128, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu, name='dis_2')
        d3 = self.conv_block(x=d2, filters=256, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu, name='dis_3')
        d4 = self.conv_block(x=d3, filters=512, kernel_size=3, strides=2, padding='SAME', train=train, actv=self.lrelu, name='dis_4')
        out = tf.layers.conv2d(d4, filters=1, kernel_size=3, strides=1, padding='SAME', name='dis_5')
        out_sigmoid = tf.sigmoid(out)
        return out_sigmoid, out

    def loss_graph_composite(self, real_crop, composite_crop, real_all, composite_all):

        G_loss_crop = tf.losses.mean_squared_error(real_crop, composite_crop) 
        G_loss_all = tf.losses.mean_squared_error(real_all, composite_all) 

        G_loss = 16 * G_loss_all

        return G_loss

    def loss_graph_restruct(self, real, fake, real_all, restruct_all):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        D_loss = D_loss_real + D_loss_fake

        G_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(fake)))
        G_loss_similarity = tf.losses.mean_squared_error(real_all, restruct_all) 

        G_loss = G_loss_fake + 32 * G_loss_similarity

        return D_loss, G_loss

    def train(self):
        real_imgs = tf.placeholder(tf.float32, self.img_shape, name='real_images')
        bg_imgs = tf.placeholder(tf.float32, self.img_shape, name='bg_images')
        composited_imgs = tf.placeholder(tf.float32, self.img_shape, name='composited_imgs')
        fg_imgs = tf.placeholder(tf.float32, [240, 320], name='fg_images')

        with tf.variable_scope('composite_generator', reuse=tf.AUTO_REUSE):
            crop_img, res_img, quantize = self.foreground_codec(real_imgs)
            composite_img, num_block = self.composite(bg_imgs, real_imgs, fg_imgs)

        with tf.variable_scope('restruct_generator'):
            restructed_img = self.quality_enhance(composited_imgs)

        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            D_y, _ = self.discriminator(real_imgs)
            D_Gy, _ = self.discriminator(restructed_img)

        # 损失  
        G_loss_com = self.loss_graph_composite(crop_img, res_img, real_imgs, composite_img)
        D_loss, G_loss_res = self.loss_graph_restruct(D_y, D_Gy, real_imgs, restructed_img)


        train_vars = tf.trainable_variables()   # 优化

        gen_vars_com = [var for var in train_vars if var.name.startswith('composite_generator')]   # 生成器变量
        gen_vars_res = [var for var in train_vars if var.name.startswith('restruct_generator')]   # 生成器变量
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]   # 判别器变量

        # 生成器与判别器作为两个网络需要分别优化
        gen_optimizer_com = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(G_loss_com, var_list=gen_vars_com)
        gen_optimizer_res = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(G_loss_res, var_list=gen_vars_res)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1).minimize(D_loss, var_list=dis_vars)

        saver = tf.train.Saver()
        
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, "/home/wulirong/video/Model_old/Final/video.model")

            for epoch in range(self.epoch_size):
                try:
                    if os.path.exists('Results/'+str(epoch)+os.sep+'Background/') is False:
                        os.makedirs('Results/'+str(epoch)+os.sep+'Background/')
                    if os.path.exists('Results/'+str(epoch)+os.sep+'Composite/') is False:
                        os.makedirs('Results/'+str(epoch)+os.sep+'Composite/')
                    if os.path.exists('Results/'+str(epoch)+os.sep+'Restruct/') is False:
                        os.makedirs('Results/'+str(epoch)+os.sep+'Restruct/')
                    if os.path.exists('Results/'+str(epoch)+os.sep+'Original/') is False:
                        os.makedirs('Results/'+str(epoch)+os.sep+'Original/')
                    if os.path.exists('Results/'+str(epoch)+os.sep+'Foreground/') is False:
                        os.makedirs('Results/'+str(epoch)+os.sep+'Foreground/')
                    file = open('Results/'+str(epoch)+os.sep+'SSIM.txt', mode='a+')

                    path = "/home/wulirong/video/dataset/train_HK"
                    videos = os.listdir(path)
                    np.random.shuffle(videos)
                    num = 0

                    for video_num in videos:
                        video_length = len(os.listdir(path + os.sep + video_num))
                        mog = cv2.createBackgroundSubtractorMOG2()
                        
                        for _ in range(3):
                            for i in range(video_length):  
                                frame = cv2.imread(path + os.sep + video_num + os.sep + str(i) + '.png')
                                img_fg = mog.apply(frame)

                        img_background = mog.getBackgroundImage()
                        img_bg = np.array(img_background) / 127.5 - 1.
                        mog = cv2.createBackgroundSubtractorMOG2()

                        for i in range(video_length):
                            frame = cv2.imread(path + os.sep + video_num + os.sep + str(i) + '.png')
                            img_fg = mog.apply(frame)
                            ret, img_fg = cv2.threshold(img_fg, 12, 255, cv2.THRESH_BINARY)
                            kernel_open_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
                            kernel_open_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
                            img_fg = cv2.morphologyEx(img_fg, cv2.MORPH_OPEN, kernel_open_1)
                            img_fg = cv2.dilate(img_fg, kernel_dilate, iterations = 1)
                            img_fg = cv2.morphologyEx(img_fg, cv2.MORPH_OPEN, kernel_open_2)
                            img_fg = np.array(img_fg) / 127.5 - 1.
                            frame = np.array(frame) / 127.5 - 1.

                            #_ = sess.run(gen_optimizer_com, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})
                            #_ = sess.run(gen_optimizer_com, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})

                            com_img = sess.run(composite_img, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})

                            _ = sess.run(dis_optimizer, feed_dict={real_imgs: frame, composited_imgs: com_img})
                            _ = sess.run(gen_optimizer_res, feed_dict={real_imgs: frame, composited_imgs: com_img})
                            _ = sess.run(gen_optimizer_res, feed_dict={real_imgs: frame, composited_imgs: com_img})
                            _ = sess.run(gen_optimizer_res, feed_dict={real_imgs: frame, composited_imgs: com_img})

                            if i % 20 == 0: 
                                loss_d = sess.run(D_loss, feed_dict={real_imgs: frame, composited_imgs: com_img})
                                loss_g = sess.run(G_loss_res, feed_dict={real_imgs: frame, composited_imgs: com_img})
                                num_block_size = sess.run(num_block, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})
                                print(datetime.now().strftime('%c'), epoch, num, i, 'D:', loss_d, 'G:', loss_g, 'N:', num_block_size)
                                #loss_g = sess.run(G_loss_com, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})
                                #num_block_size = sess.run(num_block, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})
                                #print(datetime.now().strftime('%c'), epoch, num, i, 'G:', loss_g, 'N:', num_block_size)

                            if num == self.video_size - 1:
                                com_img = sess.run(composite_img, feed_dict={real_imgs: frame, bg_imgs: img_bg, fg_imgs: img_fg})
                                res_img = sess.run(restructed_img, feed_dict={real_imgs: frame, composited_imgs: com_img})

                                com_ssim = compare_ssim(tf.cast(frame, tf.float32).eval(), tf.cast(com_img, tf.float32).eval(), win_size=5, multichannel=True)
                                res_ssim = compare_ssim(tf.cast(frame, tf.float32).eval(), tf.cast(res_img, tf.float32).eval(), win_size=5, multichannel=True)
                                file.write(str(epoch)+": "+str(com_ssim)+"  "+str(res_ssim)+"\n")
                                file.flush()

                                img_fg = (np.array(img_fg) + 1.) * 127.5
                                com_img = (np.array(com_img) + 1.) * 127.5
                                res_img = (np.array(res_img) + 1.) * 127.5
                                frame = (np.array(frame) + 1.) * 127.5

                                cv2.imwrite('Results/'+str(epoch)+os.sep+'Background/'+str(i)+'.png', img_background)
                                cv2.imwrite('Results/'+str(epoch)+os.sep+'Foreground/'+str(i)+'.png', img_fg)
                                cv2.imwrite('Results/'+str(epoch)+os.sep+'Composite/'+str(i)+'.png', com_img)
                                cv2.imwrite('Results/'+str(epoch)+os.sep+'Restruct/'+str(i)+'.png', res_img)
                                cv2.imwrite('Results/'+str(epoch)+os.sep+'Original/'+str(i)+'.png', frame)

                                #scipy.misc.imsave('Results/'+str(epoch)+os.sep+'Background/'+str(i)+'.png', img_background)
                                #scipy.misc.imsave('Results/'+str(epoch)+os.sep+'Composite/'+str(i)+'.png', com_img)
                                #scipy.misc.imsave('Results/'+str(epoch)+os.sep+'Restruct/'+str(i)+'.png', res_img)
                                #scipy.misc.imsave('Results/'+str(epoch)+os.sep+'Original/'+str(i)+'.png', frame)
                        num += 1
                    num = 0

                    if epoch % 12 == 11 :
                        if os.path.exists("/home/wulirong/video/Model/" + str(epoch) + '/') is False:
                            os.makedirs('Model/' + str(epoch) + '/')
                        model_path = os.getcwd() + os.sep + 'Model/' + str(epoch) + '/video.model'
                        saver.save(sess, model_path)

                except:
                    if os.path.exists("/home/wulirong/video/Model/Temporary/") is False:
                        os.makedirs("/home/wulirong/video/Model/Temporary/")
                    model_path = os.getcwd() + os.sep + "Model/Temporary/video.model"
                    saver.save(sess, model_path)

            if os.path.exists("/home/wulirong/video/Model/Final/") is False:
                os.makedirs("/home/wulirong/video/Model/Final/")
            model_path = os.getcwd() + os.sep + "Model/Final/video.model"
            saver.save(sess, model_path)
