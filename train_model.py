import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import data_loader
from config import get_config
from models import *

def train(generator_one, generator_two, discriminator, L1_criterion, BCE_criterion, gen_train_op1, gen_train_op2, dis_train_op1, pose_loader, config):
	if config.pretrained_path is not None:
		generator_one.load_state_dict(torch.load(os.path.join(config.pretrained_path, 'train_generator_one')))
		generator_two.load_state_dict(torch.load(os.path.join(config.pretrained_path, 'train_generator_two')))
		discriminator.load_state_dict(torch.load(os.path.join(config.pretrained_path, 'train_discriminator')))

    for epoch in range(config.epochs):
        for step, example in enumerate(pose_loader):
            [x, x_target, pose_target, mask_target] = example
            if config.use_gpu:
	            x = Variable(x.cuda())
	            x_target = Variable(x_target.cuda())
	            pose_target = Variable(pose_target.cuda())
	            mask_target = Variable(mask_target.cuda())
            
            G1 = generator_one(torch.cat([x, pose_target], dim=1))
            if step < 22000:
                PoseMaskLoss1 = L1_criterion(G1 * mask_target, x_target * mask_target)
                g_loss_1 = L1_criterion(G1, x_target) + PoseMaskLoss1
                gen_train_op1.zero_grad()
                g_loss_1.backward()
                gen_train_op1.step()
                print('Epoch: %d, Step: %d, g_loss1: %0.05f' %(epoch+1, step+1, g_loss_1))
                if step % 100 == 99:
                    torch.save(generator_one.state_dict(), os.path.join(config.checkpoint_dir, 'train_generator_one'))
                continue

            DiffMap = generator_two(torch.cat([G1, x], dim=1))
            G2 = G1 + DiffMap
            triplet = torch.cat([x_target, G2, x], dim=0)
            D_z = discriminator(triplet)
            D_z = torch.clamp(D_z, 0.0, 1.0)
            D_z_pos_x_target, D_z_neg_g2, D_z_neg_x = D_z[0], D_z[1], D_z[2]
            D_z_pos = D_z_pos_x_target
            D_z_neg = torch.cat([D_z_neg_g2, D_z_neg_x], 0)

            PoseMaskLoss1 = L1_criterion(G1 * mask_target, x_target * mask_target)
            g_loss_1 = L1_criterion(G1, x_target) + PoseMaskLoss1

            g_loss_2 = BCE_criterion(D_z_neg, torch.ones((2)).cuda())
            PoseMaskLoss2 = L1_criterion(G2 * mask_target, x_target * mask_target)
            L1Loss2 = L1_criterion(G2, x_target) + PoseMaskLoss2
            g_loss_2 += 50*L1Loss2

            gen_train_op2.zero_grad()
            g_loss_2.backward(retain_graph=True)
            gen_train_op2.step()

            d_loss = BCE_criterion(D_z_pos, torch.ones((1)).cuda())
            d_loss += BCE_criterion(D_z_neg, torch.zeros((2)).cuda())
            d_loss /= 2

            dis_train_op1.zero_grad()
            d_loss.backward()
            dis_train_op1.step()

            print('Epoch: %d, Step: %d, g_loss1: %0.05f, g_loss2: %0.05f, d_loss: %0.05f' %(epoch+1, step+1, g_loss_1, g_loss_2, d_loss))
            if step % 100 == 99:
                torch.save(generator_one.state_dict(), os.path.join(config.checkpoint_dir, 'train_generator_one'))
                torch.save(generator_two.state_dict(), os.path.join(config.checkpoint_dir, 'train_generator_two'))
                torch.save(discriminator.state_dict(), os.path.join(config.checkpoint_dir, 'train_discriminator'))


def main(config):
	if config.gpu > -1:
		os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

	generator_one = GeneratorCNN_Pose_UAEAfterResidual_256(21, config.z_num, config.repeat_num, config.hidden_num)
	generator_two = UAE_noFC_AfterNoise(6, config.repeat_num - 2, config.hidden_num)
	discriminator = DCGANDiscriminator_256(use_gpu=config.use_gpu)

	if config.use_gpu:
		generator_one.cuda()
		generator_two.cuda()
		discriminator.cuda()

	L1_criterion = nn.L1Loss()
	BCE_criterion = nn.BCELoss()

	gen_train_op1 = optim.Adam(generator_one.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
	gen_train_op2 = optim.Adam(generator_two.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
	dis_train_op1 = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))

	pose_loader = data_loader.get_loader(os.path.join(config.data_dir, 'DF_img_pose'), config.batch_size) 
	train(generator_one, generator_two, discriminator, L1_criterion, BCE_criterion, gen_train_op1, gen_train_op2, dis_train_op1, 
		pose_loader, config)


if __name__ == "__main__":
    config, unparsed = get_config()
	main(config)
