# Phu's Blog

This is a blog that is based on **[Nehalem](https://github.com/nehalist/gatsby-theme-nehalem)** Gatsby theme deployed on [Netlify](https://www.netlify.com/). Contains mostly research stuff in Machine Learning, Deep Learning, Reinforcement Learning, Computer Science, etc...

## Installation

There seems to be a problem with installation due to Node.js version. The process of running a full setup with locally running the web is

```bash
nvm install 10.16.0
nvm use 10.16.0
git clone <https://github.com/Phutoast/phu_blog.git>
cd phu_blog
yarn
yarn workspace demo develop
```

## Workflow

Warning: This is a note mainly to me on how to write the contents.

- I usually start in either [Notion.so](http://notion.so/) or [Overleaf.com](http://overleaf.com/) as all the drafts are listed within my private [Notion.so](http://notion.so/) database. The stages are defined as follows (following on the progress)
    - `Dream`: I want to write about it, but I have near 0 knowledge about it.
    - `Idea`: I have some idea on the contents, but I don't have time to write them down. (My first paper read should be done in this stage)
    - `Writing`: I know the structure of the contents and able to write the text, which includes a development prototype of "interactive" pieces that might appear, and a more thorough review of the papers.
    - `Init-Written`: I have written everything, and now I should wait for my review.
    - `Draft-Review`: After some time after I have finished my original writing, I will come back to review my draft (including interactive pieces).
    - `Draft-Pub`: I have finished my review of my draft. Correct all the grammars and contents and are ready to added to my GitHub repo.
- After I have finished the first draft, we will create a new-brach for the writing, so that I can do a pull-request.
- If I am happy with the current writing, its time to do a pull-request, which will appear in the project page, which I can do a final review~

Suppose I want to edit/add new contents for the existing one. I will have to change the status of the [Notion.so](http://notion.so/) database to `Writing` and follows the workflow above. 

*Meta: I also follow this workflow when I am creating this [README.md](http://readme.md/)*

## Creating New Page

There are two types of content I can add: a Blog Post and a Webpage.

- To create a blog post, we created a new folder within a folder `demo/content/posts`. Within the folder, we have a cover picture (usually named `cover.jpg`), and the main markdown with the same name as the folder containing the following header:

    ```markdown
    ---
    title: "TITLE HERE"
    path: "/path-here"
    tags: ["ETC"]
    featuredImage: "./cover.jpg"
    excerpt: EXCERPT OF THE CONTENT
    created: 2019-12-02
    updated: 2019-12-02
    ---

    *Cover by [@anniespratt](https://unsplash.com/@anniespratt) Thanks*
    ```

    and don't forget to credit who created the cover picture

- For a page, it is more straightforward as we have to add a new markdown to `demo/content/pages` with the following header.

    ```markdown
    ---
    title: TITLE HERE
    path: "/url-path-here"
    excerpt: EXCERPT OF THE CONTENT
    ---
    ```

    If you want to have this page on the sidebar, we can change the configuration file `theme/gatsby-config.js`

The images can be referenced *relative* to the markdown (which is useful), and the pages can be linked using a simple markdown link.

## Authors

The Theme is developed by [nehalist.io](https://nehalist.io/). Contents are written by [Phu Sakulwongtana](https://www.phublog.me/). Contact me via this [email](mailto:phusakulwongtana@gmail.com).
