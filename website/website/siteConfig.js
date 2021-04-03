/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.


const siteConfig = {
  title: 'PyTorchVideo', // Title for your website.
  tagline: 'A deep learning library for video understanding research',
  url: 'https://pytorchvideo.org', // Your website URL
  baseUrl: '/', 

  // Used for publishing and more
  projectName: 'pytorchvideo',
  organizationName: 'facebookresearch',

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {doc: 'tutorial_overview', label: 'Tutorials'},
    {href: "https://ptv-temp.readthedocs.io/en/latest/index.html", label: 'Docs'}, // TODO: Change this after the repo becomes public.
    {href: "https://github.com/facebookresearch/pytorchvideo/", label: 'GitHub'}, //TODO: Change this after repo becomes public
  ],


  /* path to images for header/footer */
  headerIcon: 'img/logo.svg',
  footerIcon: 'img/logo.svg',
  favicon: 'img/favicon.png',

  /* Colors for website */
  colors: {
    primaryColor: '#812ce5',
    secondaryColor: '#cc33cc',
  },

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'atom-one-dark',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ['https://buttons.github.io/buttons.js'],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/logo.svg',
  twitterImage: 'img/logo.svg',

};

module.exports = siteConfig;
