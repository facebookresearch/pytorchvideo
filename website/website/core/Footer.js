/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

class Footer extends React.Component {
  docUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return `${baseUrl}docs/${language ? `${language}/` : ''}${doc}`;
  }

  pageUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return baseUrl + (language ? `${language}/` : '') + doc;
  }

  render() {
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <a href={this.props.config.baseUrl} className="nav-home">
            {this.props.config.footerIcon && (
              <img
                src={this.props.config.baseUrl + this.props.config.footerIcon}
                alt={this.props.config.title}
                width="66"
                height="58"
              />
            )}
          </a>
          <div>
            <h5>Docs</h5>
            <a href={this.docUrl('about_spectrum', this.props.language)}>
              About PyTorchVideo
            </a>
            <a href={this.docUrl('getting_started_android', this.props.language)}>
              Getting Started
            </a>
            <a href={this.docUrl('contributing_android', this.props.language)}>
              Contributing
            </a>
          </div>
          <div>
            <h5>Community</h5>
            <a
              href="https://www.facebook.com/libspectrum"
              target="_blank"
              rel="noreferrer noopener">
              Facebook
            </a>
            <a
              href="https://twitter.com/libspectrum"
              target="_blank"
              rel="noreferrer noopener">
              Twitter
            </a>
          </div>
          <div>
            <h5>Legal</h5>
            <a
              href="https://opensource.facebook.com/legal/terms"
              target="_blank"
              rel="noreferrer noopener">
              Terms of Use
            </a>
            <a
              href="https://opensource.facebook.com/legal/data-policy"
              target="_blank"
              rel="noreferrer noopener">
              Data Policy
            </a>
            <a
              href="https://opensource.facebook.com/legal/cookie-policy"
              target="_blank"
              rel="noreferrer noopener">
              Cookie Policy
            </a>
          </div>
          <div>
            <h5>More</h5>
            <a href={this.props.config.repoUrl}>GitHub</a>
            <a
              className="github-button"
              href={this.props.config.repoUrl}
              data-icon="octicon-star"
              data-count-href="/facebookincubator/spectrum/stargazers"
              data-show-count="true"
              data-count-aria-label="# stargazers on GitHub"
              aria-label="Star this project on GitHub">
              Star
            </a>
          </div>
        </section>

        <a
          href="https://code.facebook.com/projects/"
          target="_blank"
          rel="noreferrer noopener"
          className="fbOpenSource">
          <img
            src={`${this.props.config.baseUrl}img/oss_logo.png`}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">{this.props.config.copyright}</section>
      </footer>
    );
  }
}

module.exports = Footer;