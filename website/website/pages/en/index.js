/**
 * Copyright (c) 2021-present, Facebook, Inc.
**/

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;
const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;
class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const ProjectTitle = props => (
      <h2 className="projectTitle">
        <small>{props.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={baseUrl + 'img/logo_white.svg'} />
        <div className="inner">
          <ProjectTitle tagline={siteConfig.tagline} title={siteConfig.title} />
          <PromoSection>
            <Button href={'#quickstart'}>Get Started</Button>
            <Button href={docUrl('tutorial_overview.html')}>Tutorials</Button>
            <Button href={"https://github.com/facebookresearch/pytorchvideo"}>GitHub</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/placeholder.png`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );

    const pre = '```';

    const codeExample = `${pre}python
from pytorchvideo import foo
from pytorchvideo.models import bar
    `;
    const install = `${pre}bash
pip install pytorchvideo
    `;

    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <strong>Install pytorchvideo  </strong> (Confirm requirements following the instructions <a href="https://github.com/facebookresearch/pytorchvideo/blob/master/INSTALL.md">here</a>)
              <MarkdownBlock>{install}</MarkdownBlock>
            </li>
            <li>
              <strong>Try Video classification with Model Zoo  </strong>
              <MarkdownBlock>{codeExample}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const UseCases = () => (
      <div className="productShowcaseSection" style={{textAlign: 'center'}}>
        <h1>Some use cases</h1>
        <div className="column">
          <div className="row">
            
            <br></br>
            <h3>Detection (Add GIF)</h3>
          </div>
          <div className="row">
            
            <br></br>
            <h3>Tracking (Add GIF)</h3>
          </div>
          <div className="row">
            
            <br></br>
            <h3>Classification (Add GIF)</h3>
          </div>
        </div>
      </div>
    );

    const Features = () => (
      <div className="productShowcaseSection" style={{textAlign: 'center'}}>
        <Block layout="fourColumn">
          {[
            {
              content:
                'Built using PyTorch. Makes it easy to use all the PyTorch-ecosystem components.',
              image: `${baseUrl}img/pytorch.svg`,
              imageAlign: 'top',
              title: 'Based on PyTorch',
            },
            {
              content:
                'Variety of state of the art pretrained video models and their associated benchmarks that are ready to use.',
              image: `${baseUrl}img/modelzoo.svg`,
              imageAlign: 'top',
              title: 'Reproducible Model Zoo',
            },
            // {
            //   content:
            //     'Variety of benchmark tasks available to evaluate the models.',
            //   image: `${baseUrl}img/reproducible.svg`,
            //   imageAlign: 'top',
            //   title: 'Reproducible Benchmarks',
            // },
            {
              content:
                'Video-focused fast and efficient components that are easy to use. Supports accelerated inference on hardware.',
              image: `${baseUrl}img/efficient.svg`,
              imageAlign: 'top',
              title: 'Efficient Video Components',
            },
          ]}
        </Block>
      </div>
    );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;