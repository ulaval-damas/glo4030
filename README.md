# glo4030.github.io

Site web du cours GLO-4030/GLO-7030

### Développer localement

Pour travailler sur le site web du cours sans avoir à le déployer, il faut d'abord installer les dépendances:


``` bash
sudo apt install ruby-full
echo 'export PATH=$PATH:~/.gem/ruby/2.4.0/bin >> ~/.bashrc
source ~/.bashrc

gem install jekyll bundler github-pages
```

Une fois les dépendances installée, il suffit d'aller dans le repository du site web et lancer:

```bash
bundle exec jekyll serve --watch
```

Cette commande relance le build du site web dès qu'il y a une modification.
