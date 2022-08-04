def clean_caption(caption, classnames, truncate):
    result = [word for word in caption.split() if word not in classnames][:truncate]
    while len(result) < truncate:
        result.append("pad")
        print(f"Padded {caption}")
    return result


caption = "This is a caption"
classnames = ["is"]
truncate = 5

a = clean_caption(caption, classnames, truncate)
print(a)