import os
import genboxes as gb
import hydra
import torch

@hydra.main(config_path="configs", config_name="default")
def main(cfg):
    save = cfg.settings.save
    if hasattr(cfg.settings, "gen_in_bulk"):
        gen_in_bulk = getattr(cfg.settings, "gen_in_bulk")
    else:
        gen_in_bulk = False
    if save:
        # prepare data saving
        save_dir = os.path.expanduser(cfg.settings.save_folder)
        if not os.path.exists(save_dir):
            print("Creating dir:", save_dir)
            os.makedirs(save_dir)
        else:
            print("Directory already exists:", save_dir)
        modes_dir = {"train": os.path.join(save_dir, "train"),
                    "test": os.path.join(save_dir, "test")}
        os.makedirs(modes_dir["train"], exist_ok=True)
        os.makedirs(modes_dir["test"], exist_ok=True)
        # Create val directory in case it is specified
        if hasattr(cfg.settings, "val"):
            modes_dir["val"] = os.path.join(save_dir, "val")
            os.makedirs(modes_dir["val"], exist_ok=True)
    if not gen_in_bulk:
        gb_train = gb.BoxesDataset(cfg, 'train')
        gb_test = gb.BoxesDataset(cfg, 'test')
        data_train = gb_train.get_data()
        data_test = gb_test.get_data()
        if cfg.settings.visu:
            gb_train.visualize()
            gb_test.visualize()
        if save:
            torch.save(data_train, os.path.join(modes_dir["train"],'grid300mm_data.pt'))
            torch.save(data_test, os.path.join(modes_dir["test"],'grid300mm_data.pt'))
    else:
        default_modes = ["train", "val", "test"]
        for mode in default_modes:
            if hasattr(cfg.settings, mode):
                nb_scenes = getattr(cfg.settings, mode)
                if save:
                    mode_dir = modes_dir[mode]
                for i in range(nb_scenes):
                    gb_scene = gb.BoxesDataset(cfg, mode)
                    gb_data = gb_scene.get_data()
                    if save:
                        torch.save(gb_data,
                                   os.path.join(mode_dir, f'grid300mm_data_{i}.pt'))
                if cfg.settings.visu:
                    gb_scene.visualize()


if __name__ == '__main__':
    main()