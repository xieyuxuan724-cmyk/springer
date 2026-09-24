// Each skin selects a colour atlas; the other material maps are shared.
// primary/accent are UI colours sampled from the atlas.
export const SKINS = [
  {
    id: 'white',
    name: 'Polar White',
    primary: '#dbdbdb',
    accent: '#f5f7f9',
    texture: 'beni_white.png',
  },
  {
    id: 'black',
    name: 'Obsidian Black',
    primary: '#6f757c',
    accent: '#b9c0c7',
    texture: 'beni_Black.jpg',
  },
  {
    id: 'red',
    name: 'Lava Red',
    primary: '#e24647',
    accent: '#ffb9b3',
    texture: 'beni_red.jpg',
  },
  {
    id: 'blue',
    name: 'Ocean Blue',
    primary: '#4cbeea',
    accent: '#c6ecff',
    texture: 'beni_Blue.jpg',
  },
  {
    id: 'green',
    name: 'Neon Green',
    primary: '#98e94c',
    accent: '#ddffb8',
    texture: 'beni_Green.jpg',
  },
  {
    id: 'yellow',
    name: 'Racing Orange',
    primary: '#ea934c',
    accent: '#ffdcb3',
    texture: 'beni_yellow.jpg',
  },
  {
    id: 'purple',
    name: 'Cosmic Purple',
    primary: '#e84bdf',
    accent: '#ffc4fb',
    texture: 'beni_Purple.jpg',
  },
  {
    id: 'camo',
    name: 'Desert Camo',
    primary: '#a08a63',
    accent: '#ddcdae',
    texture: 'beni_micai.jpg',
  },
];
export const getSkin = (id) => SKINS.find((s) => s.id === id) ?? SKINS[0];
export const skinTextureUrl = (skin) => '/robot/textures/' + skin.texture;
export function skinColor(skin, geom) {
  const brightness = Math.max(...geom.color);
  return brightness < 0.3
    ? '#202932'
    : brightness < 0.8
      ? skin.primary
      : skin.accent;
}
